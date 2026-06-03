#!/usr/bin/env python
"""Avalia um checkpoint fine-tunado de RTMPose-X no val split do 3DSP.

Carrega o modelo via MMPose, aplica letterboxing 100×100 → 288×384,
coleta predições no val split e calcula as 4 métricas do projeto.

Uso:
    python scripts/evaluate.py \\
        --checkpoint results/checkpoints/cenario_C/best_PCK.pth \\
        --config configs/cenario_c.py \\
        [--split val] \\
        [--data-dir data] \\
        [--split-config configs/split.json] \\
        [--output-dir results/tables] \\
        [--device cuda]

Saída:
    - Tabela de métricas no terminal
    - JSON salvo em output_dir/finetuned_<cenario>_<split>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from football_orient_pose.evaluation import (
    compute_mpjpe_2d,
    compute_oks,
    compute_pck,
    compute_pdj,
)
from football_orient_pose.utils.data_io import load_clip_image, load_keypoints_2d


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Avalia checkpoint fine-tunado no 3DSP")
    p.add_argument("--checkpoint", required=True, help="Path para o .pth do checkpoint")
    p.add_argument(
        "--config", default=None,
        help="Path para o config MMPose. Se omitido, infere do nome do checkpoint.",
    )
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--split-config", type=Path, default=Path("configs/split.json"))
    p.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def _infer_config(checkpoint_path: str) -> str:
    """Infere o config do cenário pelo path do checkpoint."""
    p = Path(checkpoint_path)
    for part in p.parts:
        part_lower = part.lower()
        for cenario in ("a", "b", "c", "d"):
            if f"cenario_{cenario}" in part_lower:
                config = Path("configs") / f"cenario_{cenario}.py"
                if config.exists():
                    return str(config)
    raise ValueError(
        f"Não foi possível inferir o config a partir de '{checkpoint_path}'. "
        "Passe --config explicitamente."
    )


def run_evaluation(
    checkpoint_path: str,
    config_path: str,
    split: str,
    data_dir: Path,
    split_config: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    """Avalia o modelo e retorna dicionário com todas as métricas.

    Pode ser chamado pelo train.py para calcular Δ PCK entre fases.
    """
    from mmpose.apis import init_model, inference_topdown

    print(f"\nCarregando modelo: {checkpoint_path}")
    model = init_model(config_path, checkpoint_path, device=device)
    model.eval()

    split_data = json.loads(split_config.read_text())
    if split not in split_data:
        raise ValueError(f"Split '{split}' não encontrado em {split_config}")
    clip_ids: list[str] = split_data[split]

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total_frames = len(clip_ids) * 20

    try:
        from tqdm import tqdm
        frames_iter = tqdm(range(total_frames), desc="Inferência", unit="frame")
    except ImportError:
        frames_iter = range(total_frames)

    frame_counter = 0
    for clip_id in clip_ids:
        clip_dir = data_dir / "train" / clip_id
        for frame_idx in range(1, 21):
            img = load_clip_image(clip_dir, frame_idx)
            H, W = img.shape[:2]

            # Passa a bbox cobrindo o crop inteiro — MMPose aplica letterboxing
            # via TopdownAffine e devolve keypoints no espaço do crop original.
            results = inference_topdown(
                model,
                img,
                bboxes=np.array([[0.0, 0.0, float(W), float(H)]]),
                bbox_format="xyxy",
            )

            if results and len(results[0].pred_instances.keypoints) > 0:
                kps = np.asarray(
                    results[0].pred_instances.keypoints[0], dtype=np.float32
                )  # (17, 2) em coordenadas do crop
            else:
                kps = np.zeros((17, 2), dtype=np.float32)

            gt_kps = load_keypoints_2d(
                clip_dir / "posture" / f"{frame_idx:03d}.json"
            )  # (17, 2)

            predictions.append(kps)
            targets.append(gt_kps)
            frame_counter += 1
            if hasattr(frames_iter, "update"):
                frames_iter.update(1)

    if hasattr(frames_iter, "close"):
        frames_iter.close()

    pred = np.asarray(predictions, dtype=np.float32)  # (N, 17, 2)
    gt = np.asarray(targets, dtype=np.float32)         # (N, 17, 2)

    pdj_r = compute_pdj(pred, gt, threshold=0.5)
    pck_r = compute_pck(pred, gt, threshold=0.2)
    oks_r = compute_oks(pred, gt)
    mpjpe_r = compute_mpjpe_2d(pred, gt)

    # Extrai cenário do checkpoint path para o filename de saída
    checkpoint_stem = Path(checkpoint_path).stem
    ckpt_parent = Path(checkpoint_path).parent.name

    result = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "config": config_path,
        "split": split,
        "n_clips": len(clip_ids),
        "n_frames": int(pred.shape[0]),
        "pdj_05": round(float(pdj_r.global_score), 6),
        "pck_02": round(float(pck_r.global_score), 6),
        "oks": round(float(oks_r.global_oks), 6),
        "oks_ap50": round(float(oks_r.ap50), 6),
        "oks_ap75": round(float(oks_r.ap75), 6),
        "oks_map": round(float(oks_r.ap), 6),
        "mpjpe_2d_px": round(float(mpjpe_r.global_mpjpe), 4),
        "pdj_per_group": {k: round(v, 6) for k, v in pdj_r.per_group.items()},
        "pck_per_group": {k: round(v, 6) for k, v in pck_r.per_group.items()},
        "mpjpe_per_group": {k: round(v, 4) for k, v in mpjpe_r.per_group.items()},
    }

    _print_table(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"finetuned_{ckpt_parent}_{split}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"Resultados salvos em {out_file}")

    return result


def _print_table(r: dict) -> None:
    w = 55
    print(f"\n{'='*w}")
    print(f"  Checkpoint: {Path(r['checkpoint']).name}")
    print(f"  Split: {r['split']}  |  Clips: {r['n_clips']}  |  Frames: {r['n_frames']}")
    print(f"{'='*w}")
    print(f"  PDJ@0.5     {r['pdj_05']*100:>8.2f}%")
    print(f"  PCK@0.2     {r['pck_02']*100:>8.2f}%")
    print(f"  OKS         {r['oks']*100:>8.2f}%")
    print(f"  AP50        {r['oks_ap50']*100:>8.2f}%")
    print(f"  AP75        {r['oks_ap75']*100:>8.2f}%")
    print(f"  mAP         {r['oks_map']*100:>8.2f}%")
    print(f"  MPJPE-2D    {r['mpjpe_2d_px']:>8.2f} px")
    print(f"{'='*w}")
    print("\n  Por grupo anatômico (PCK@0.2):")
    for group, score in r["pck_per_group"].items():
        print(f"    {group:<14} {score*100:>6.2f}%")
    print(f"{'='*w}\n")


def main() -> None:
    args = _parse_args()

    config_path = args.config or _infer_config(args.checkpoint)
    if not Path(config_path).exists():
        print(f"[ERRO] Config não encontrado: {config_path}")
        sys.exit(1)
    if not Path(args.checkpoint).exists():
        print(f"[ERRO] Checkpoint não encontrado: {args.checkpoint}")
        sys.exit(1)

    run_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=config_path,
        split=args.split,
        data_dir=args.data_dir,
        split_config=args.split_config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
