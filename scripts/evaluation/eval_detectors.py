#!/usr/bin/env python
"""Benchmark dos 4 detectores de pessoa no GT humano dos examples (Épico #113, task #115).

Roda um detector em cada frame de ``data/clips/examples/<clip>/img/NNN.jpg``, compara com o GT
de bbox (``data/annotations/examples_bbox``, via ``roboflow_io``) e calcula, com pycocotools:
mAP@[.5:.95], AP50/AP75, AP/AR por tamanho (small/medium/large) + precision/recall/F1 no ponto de
operação. Salva um JSON por detector em ``results/tables/detector_<nome>.json``.

O estágio caro (rodar o modelo) é separado do barato (métricas): use ``--save-predictions`` para
cachear as caixas em ``results/detections/`` e ``--from-predictions`` para recomputar métricas sem
GPU.

Uso:
    python scripts/evaluation/eval_detectors.py --detector yolo26 --device cuda
    python scripts/evaluation/eval_detectors.py --detector faster-rcnn --save-predictions
    python scripts/evaluation/eval_detectors.py --detector yolo26 \\
        --from-predictions results/detections/yolo26_examples.json
    python scripts/evaluation/eval_detectors.py --detector cascade-rcnn \\
        --config <cfg.py> --checkpoint <ckpt.pth>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from football_orient_pose.detection import (
    CascadeRCNNDetector,
    Detector,
    FasterRCNNDetector,
    RetinaNetDetector,
    YOLO26Detector,
    detections_to_arrays,
)
from football_orient_pose.evaluation.detection_metrics import (
    build_coco_dt,
    build_coco_gt,
    cocoeval_stats,
    precision_recall_f1,
)
from football_orient_pose.utils.roboflow_io import load_examples_bbox_gt
from football_orient_pose.utils.viz import draw_boxes

DETECTORS = ("yolo26", "faster-rcnn", "retinanet", "cascade-rcnn")

# preds: {clip_id: {frame_idx: (boxes (N,4) float32, scores (N,) float32)}}
Predictions = dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]


def _build_detector(name: str, conf: float, device: str | None, args) -> Detector:
    if name == "yolo26":
        return YOLO26Detector(weights=args.weights, conf_threshold=conf, device=device)
    if name == "faster-rcnn":
        return FasterRCNNDetector(conf_threshold=conf, device=device)
    if name == "retinanet":
        return RetinaNetDetector(conf_threshold=conf, device=device)
    if name == "cascade-rcnn":
        if not args.config or not args.checkpoint:
            raise SystemExit("cascade-rcnn exige --config e --checkpoint do model zoo do mmdet")
        return CascadeRCNNDetector(args.config, args.checkpoint, conf_threshold=conf, device=device)
    raise ValueError(f"detector desconhecido: {name}")


def predict_clips(
    detector: Detector, data_root: Path, gt: dict[str, dict[int, np.ndarray]]
) -> Predictions:
    """Roda o detector em cada frame presente no GT. Devolve as predições por clip/frame."""
    preds: Predictions = {}
    for clip_id in sorted(gt):
        preds[clip_id] = {}
        for frame_idx in sorted(gt[clip_id]):
            img_path = data_root / clip_id / "img" / f"{frame_idx:03d}.jpg"
            image = cv2.imread(str(img_path))
            if image is None:
                raise FileNotFoundError(f"frame não encontrado: {img_path}")
            boxes, scores = detections_to_arrays(detector.detect(image))
            preds[clip_id][frame_idx] = (boxes, scores)
    return preds


def evaluate_detector(
    name: str, preds: Predictions, gt: dict[str, dict[int, np.ndarray]], conf: float
) -> dict:
    """Calcula COCOeval + PR/F1 agregado sobre todos os frames. Devolve o dict de resultado."""
    coco_gt, id_map = build_coco_gt(gt)
    stats = cocoeval_stats(coco_gt, build_coco_dt(preds, id_map))

    tp = fp = fn = n_pred = n_gt = n_frames = 0
    for clip_id in gt:
        for frame_idx, gt_boxes in gt[clip_id].items():
            boxes, scores = preds.get(clip_id, {}).get(
                frame_idx, (np.empty((0, 4), np.float32), np.empty((0,), np.float32))
            )
            r = precision_recall_f1(boxes, scores, gt_boxes, iou_thr=0.5, score_thr=conf)
            tp += r["tp"]
            fp += r["fp"]
            fn += r["fn"]
            n_pred += len(boxes)
            n_gt += len(gt_boxes)
            n_frames += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    result = {
        "detector": name,
        "n_clips": len(gt),
        "n_frames": n_frames,
        "n_gt": n_gt,
        "n_pred": n_pred,
        "conf_threshold": conf,
        **stats,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    return result


def save_predictions(preds: Predictions, path: Path) -> None:
    serial = {
        clip: {
            str(f): {"boxes": b.tolist(), "scores": s.tolist()} for f, (b, s) in frames.items()
        }
        for clip, frames in preds.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serial), encoding="utf-8")


def load_predictions(path: Path) -> Predictions:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        clip: {
            int(f): (
                np.asarray(v["boxes"], dtype=np.float32).reshape(-1, 4),
                np.asarray(v["scores"], dtype=np.float32).reshape(-1),
            )
            for f, v in frames.items()
        }
        for clip, frames in raw.items()
    }


def _save_viz(name: str, data_root: Path, gt, preds: Predictions, n: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for clip_id in sorted(gt):
        for frame_idx in sorted(gt[clip_id]):
            if saved >= n:
                return
            image = cv2.imread(str(data_root / clip_id / "img" / f"{frame_idx:03d}.jpg"))
            if image is None:
                continue
            image = draw_boxes(image, gt[clip_id][frame_idx], color=(0, 255, 0))  # GT verde
            boxes, _ = preds.get(clip_id, {}).get(frame_idx, (np.empty((0, 4)), None))
            image = draw_boxes(image, boxes, color=(0, 0, 255))  # predição vermelho
            cv2.imwrite(str(out_dir / f"{name}_{clip_id}_{frame_idx:03d}.png"), image)
            saved += 1


def _ar(value: float) -> str:
    """Formata AR: COCOeval devolve -1 quando não há objetos daquele tamanho → 'n/a'."""
    return f"{value * 100:>8.2f}%" if value >= 0 else f"{'n/a':>9}"


def _print_table(r: dict) -> None:
    w = 52
    print(f"\n{'=' * w}")
    print(f"  Detector: {r['detector']}")
    print(
        f"  Clips: {r['n_clips']} | Frames: {r['n_frames']} | "
        f"GT: {r['n_gt']} | Pred: {r['n_pred']}"
    )
    print(f"{'=' * w}")
    print(f"  mAP@[.5:.95] {r['mAP'] * 100:>8.2f}%")
    print(f"  AP50         {r['AP50'] * 100:>8.2f}%")
    print(f"  AP75         {r['AP75'] * 100:>8.2f}%")
    print(f"{'-' * w}")
    print(f"  AR small     {_ar(r['AR_small'])}   (jogador distante)")
    print(f"  AR medium    {_ar(r['AR_medium'])}")
    print(f"  AR large     {_ar(r['AR_large'])}   (jogador perto)")
    print(f"{'-' * w}")
    print(f"  Precision    {r['precision'] * 100:>8.2f}%  (conf>={r['conf_threshold']}, IoU>=.5)")
    print(f"  Recall       {r['recall'] * 100:>8.2f}%")
    print(f"  F1           {r['f1'] * 100:>8.2f}%")
    print(f"{'=' * w}\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark de detectores de pessoa (Épico #113)")
    p.add_argument("--detector", required=True, choices=DETECTORS)
    p.add_argument("--data-root", type=Path, default=Path("data/clips/examples"))
    p.add_argument("--gt-root", type=Path, default=Path("data/annotations/examples_bbox"))
    p.add_argument("--device", default=None, help="cpu|cuda|cuda:0 (default: auto)")
    p.add_argument("--conf", type=float, default=0.3, help="confidence threshold")
    p.add_argument(
        "--weights", default="yolo26n.pt",
        help="só yolo26: variante do peso (yolo26n/s/m/l/x.pt). Maior = mais justo vs ResNet50.",
    )
    p.add_argument("--out", type=Path, default=Path("results/tables"))
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--from-predictions", type=Path, default=None, help="recomputa do cache")
    p.add_argument("--viz", type=int, default=0, help="salva N frames com GT(verde)×pred(vermelho)")
    p.add_argument("--config", default=None, help="cascade-rcnn: config mmdet")
    p.add_argument("--checkpoint", default=None, help="cascade-rcnn: checkpoint mmdet")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    gt = load_examples_bbox_gt(args.gt_root)

    if args.from_predictions:
        print(f"Carregando predições de {args.from_predictions}")
        preds = load_predictions(args.from_predictions)
    else:
        detector = _build_detector(args.detector, args.conf, args.device, args)
        print(f"Rodando {detector.name} em {sum(len(v) for v in gt.values())} frames...")
        preds = predict_clips(detector, args.data_root, gt)
        if args.save_predictions:
            pred_path = Path("results/detections") / f"{args.detector}_examples.json"
            save_predictions(preds, pred_path)
            print(f"Predições salvas em {pred_path}")

    result = evaluate_detector(args.detector, preds, gt, args.conf)
    if args.detector == "yolo26":
        result["weights"] = args.weights  # registra a variante (nano/x/...) no JSON
    _print_table(result)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"detector_{args.detector}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Resultados salvos em {out_file}")

    if args.viz:
        _save_viz(args.detector, args.data_root, gt, preds, args.viz, Path("results/detections"))
        print(f"Viz salvas em results/detections/ ({args.viz} frame(s))")


if __name__ == "__main__":
    main()
