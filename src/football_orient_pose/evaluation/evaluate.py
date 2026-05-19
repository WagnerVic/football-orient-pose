"""CLI de avaliação batch: roda um estimador em um split e calcula todas as métricas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from football_orient_pose.estimators import RTMPoseEstimator
from football_orient_pose.evaluation import (
    compute_mpjpe_2d,
    compute_oks,
    compute_pck,
    compute_pdj,
    joint_detection_report,
)
from football_orient_pose.utils.data_io import load_clip_image, load_keypoints_2d


def _build_estimator(model_name: str, device: str):
    if model_name == "rtmpose":
        return RTMPoseEstimator(device=device)
    raise ValueError(f"Modelo desconhecido: '{model_name}'. Opções: rtmpose")


def _run_inference(estimator, clip_ids: list[str], data_dir: Path):
    from tqdm import tqdm

    predictions, targets = [], []
    total_frames = len(clip_ids) * 20
    with tqdm(total=total_frames, desc=f"Inferência ({estimator.name})", unit="frame") as pbar:
        for clip_id in clip_ids:
            clip_dir = data_dir / "train" / clip_id
            for frame_idx in range(1, 21):
                image = load_clip_image(clip_dir, frame_idx)
                predictions.append(estimator.predict_h3wb(image))
                targets.append(
                    load_keypoints_2d(clip_dir / "posture" / f"{frame_idx:03d}.json")
                )
                pbar.update(1)
    return np.asarray(predictions, np.float32), np.asarray(targets, np.float32)


def _print_table(result: dict) -> None:
    w = 50
    print(f"\n{'='*w}")
    print(f"  Model: {result['model']}  |  Split: {result['split']}")
    print(f"  Clips: {result['n_clips']}  |  Frames: {result['n_frames']}")
    print(f"{'='*w}")
    print(f"  PDJ@0.5     {result['pdj']['global']*100:>7.2f}%")
    print(f"  PCK@0.2     {result['pck']['global']*100:>7.2f}%")
    print(f"  OKS         {result['oks']['global_oks']*100:>7.2f}%")
    print(f"  AP50        {result['oks']['ap50']*100:>7.2f}%")
    print(f"  AP75        {result['oks']['ap75']*100:>7.2f}%")
    print(f"  mAP         {result['oks']['ap']*100:>7.2f}%")
    print(f"  MPJPE-2D    {result['mpjpe_2d']['global_px']:>7.2f} px")
    print(f"  F1-macro    {result['f1_macro']*100:>7.2f}%")
    print(f"{'='*w}\n")


def evaluate(
    model_name: str,
    split: str,
    data_dir: Path,
    split_config: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    split_data = json.loads(split_config.read_text())
    if split not in split_data:
        raise ValueError(f"Split '{split}' não encontrado em {split_config}")
    clip_ids: list[str] = split_data[split]

    estimator = _build_estimator(model_name, device)
    pred, gt = _run_inference(estimator, clip_ids, data_dir)

    pdj = compute_pdj(pred, gt, threshold=0.5)
    pck = compute_pck(pred, gt, threshold=0.2)
    oks = compute_oks(pred, gt)
    mpjpe = compute_mpjpe_2d(pred, gt)
    det = joint_detection_report(pred, gt, threshold=0.5)

    result = {
        "model": model_name,
        "split": split,
        "n_clips": len(clip_ids),
        "n_frames": int(pred.shape[0]),
        "pdj": {"global": round(float(pdj.global_score), 6), "per_group": pdj.per_group},
        "pck": {"global": round(float(pck.global_score), 6), "per_group": pck.per_group},
        "oks": {
            "global_oks": round(float(oks.global_oks), 6),
            "ap": round(float(oks.ap), 6),
            "ap50": round(float(oks.ap50), 6),
            "ap75": round(float(oks.ap75), 6),
        },
        "mpjpe_2d": {
            "global_px": round(float(mpjpe.global_mpjpe), 4),
            "per_group": {k: round(v, 4) for k, v in mpjpe.per_group.items()},
        },
        "f1_macro": round(float(det.f1_macro), 6),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{model_name}_{split}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"Resultados salvos em {out_file}")
    _print_table(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia estimador de pose no 3DSP")
    parser.add_argument("--model", default="rtmpose", choices=["rtmpose"])
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--data-dir", type=Path, default=Path("data/3dsp"))
    parser.add_argument("--split-config", type=Path, default=Path("configs/split.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        split=args.split,
        data_dir=args.data_dir,
        split_config=args.split_config,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
