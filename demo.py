#!/usr/bin/env python
"""Run the full pipeline (detection -> tight crop -> pose -> reprojection) on a video or image.

Every player detected by YOLO26x is cropped, passed to the pose estimator and drawn back on the
frame. Works on CPU with the zero-shot RTMPose-X (ONNX, downloaded on first use); the fine-tuned
checkpoint needs the MMPose environment (``make finetuning-env`` or the Docker image).

Examples:
    python demo.py --input data/examples/test_00001.mp4
    python demo.py --input frame.jpg --output frame_pose.jpg
    python demo.py --input match.mp4 --pose finetuned --checkpoint d_full.pth --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from football_orient_pose.detection import YOLO26Detector, detections_to_arrays  # noqa: E402
from football_orient_pose.pipeline import pose_all  # noqa: E402
from football_orient_pose.utils.keypoint_mapping import H3WB17_NAMES  # noqa: E402
from football_orient_pose.utils.viz import draw_boxes, draw_skeleton  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_pose_estimator(kind: str, device: str, checkpoint: str | None, config: str | None):
    from football_orient_pose.estimators.rtmpose import RTMPoseEstimator

    if kind == "rtmpose":
        # rtmlib/ONNX accepts only "cpu" or "cuda".
        return RTMPoseEstimator(device="cuda" if device == "cuda" else "cpu")
    if not checkpoint:
        raise SystemExit("--pose finetuned requires --checkpoint <path to .pth>")
    return RTMPoseEstimator.from_checkpoint(checkpoint, config_path=config, device=device)


def annotate(frame: np.ndarray, detector, pose, min_box_height: float, show_boxes: bool):
    """Returns the annotated frame and the per-player results as JSON-serialisable dicts."""
    boxes, scores = detections_to_arrays(detector.detect(frame))
    results = pose_all(frame, boxes, pose, min_box_height=min_box_height)

    vis = draw_boxes(frame, boxes, color=(160, 160, 160), thickness=1) if show_boxes else frame
    players = []
    for res in results:
        vis = draw_skeleton(vis, res.keypoints_frame)
        players.append({
            "box_xyxy": [round(float(v), 1) for v in res.finisher_box],
            "keypoints_xy": [[round(float(x), 1), round(float(y), 1)]
                             for x, y in res.keypoints_frame],
        })
    return vis, players


def run_image(path: Path, output: Path, detector, pose, args) -> list[dict]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"could not read image: {path}")
    vis, players = annotate(frame, detector, pose, args.min_box_height, not args.no_boxes)
    cv2.imwrite(str(output), vis)
    return [{"frame": 0, "players": players}]


def run_video(path: Path, output: Path, detector, pose, args) -> list[dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames) if total > 0 else args.max_frames
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    records, start = [], time.perf_counter()
    try:
        while not args.max_frames or len(records) < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            vis, players = annotate(frame, detector, pose, args.min_box_height, not args.no_boxes)
            writer.write(vis)
            records.append({"frame": len(records), "players": players})
            n = len(records)
            if n % 10 == 0 or n == total:
                rate = n / (time.perf_counter() - start)
                print(f"  frame {n}/{total or '?'}  ({rate:.1f} fps, {len(players)} players)")
    finally:
        cap.release()
        writer.release()
    return records


def default_output(input_path: Path, is_image: bool) -> Path:
    suffix = input_path.suffix if is_image else ".mp4"
    return Path("outputs") / f"{input_path.stem}_pose{suffix}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Soccer player pose estimation on a broadcast video or image.")
    p.add_argument("--input", type=Path, required=True, help="video (.mp4, ...) or image")
    p.add_argument("--output", type=Path, default=None,
                   help="annotated output (default: outputs/<name>_pose.<ext>)")
    p.add_argument("--pose", choices=["rtmpose", "finetuned"], default="rtmpose",
                   help="rtmpose = zero-shot RTMPose-X (ONNX); finetuned = MMPose checkpoint")
    p.add_argument("--checkpoint", default=None, help="fine-tuned .pth (with --pose finetuned)")
    p.add_argument("--config", default=None,
                   help="MMPose config for the checkpoint (inferred from its path if omitted)")
    p.add_argument("--detector-weights", default="yolo26x.pt",
                   help="Ultralytics weights; downloaded automatically if missing")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--min-box-height", type=float, default=40.0,
                   help="skip detections shorter than this (px): distant players give poor crops")
    p.add_argument("--max-frames", type=int, default=None, help="process only the first N frames")
    p.add_argument("--json", action="store_true",
                   help="also save boxes and H3WB-17 keypoints to <output>.json")
    p.add_argument("--no-boxes", action="store_true", help="draw skeletons only")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}")
    if args.device == "auto":
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    is_image = args.input.suffix.lower() in IMAGE_SUFFIXES
    output = args.output or default_output(args.input, is_image)
    output.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLO26Detector(weights=args.detector_weights, device=args.device)
    pose = build_pose_estimator(args.pose, args.device, args.checkpoint, args.config)
    print(f"detector: {args.detector_weights} | pose: {pose.name} | device: {args.device}")

    runner = run_image if is_image else run_video
    records = runner(args.input, output, detector, pose, args)
    print(f"saved {output}")

    if args.json:
        json_path = output.with_suffix(".json")
        payload = {"keypoint_format": "H3WB-17", "keypoint_names": H3WB17_NAMES,
                   "source": str(args.input), "frames": records}
        json_path.write_text(json.dumps(payload))
        print(f"saved {json_path}")


if __name__ == "__main__":
    main()
