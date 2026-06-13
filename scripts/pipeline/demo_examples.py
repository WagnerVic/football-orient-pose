#!/usr/bin/env python
"""Pipeline ponta-a-ponta nos examples (Épico #126): detecta → finalizador → crop → pose → viz.

Para cada clip de `data/clips/examples`, roda o pipeline em cada frame inteiro usando o YOLO26x
(detector vencedor) + um estimador de pose, e produz DOIS resultados:
- **showcase** em `results/pipeline/<clip>/` (frame + caixa + esqueleto; e o crop + esqueleto);
- **estágio formal** em `data/crops/examples/<clip>/` (crops justos + crop_params + info.ini),
  insumo da anotação de keypoints (US #109).

O finalizador é de graça nos examples: vem do `shooter_tracklet_id`/`gt.txt` do 3DSP (clip
correspondente em `data/test/`), usado como caixa de referência (casa por IoU com a detecção).

Uso:
    python scripts/pipeline/demo_examples.py --pose rtmpose --device cuda
    python scripts/pipeline/demo_examples.py --clips example_01 --pose rtmpose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from football_orient_pose.crops_io import write_crop_clip
from football_orient_pose.detection import YOLO26Detector, detections_to_arrays
from football_orient_pose.pipeline import crop_and_pose, track_finisher
from football_orient_pose.utils.data_io import load_clip_info, load_finisher_boxes
from football_orient_pose.utils.viz import draw_boxes, draw_skeleton


def _build_pose(name: str, device: str):
    if name == "rtmpose":
        from football_orient_pose.estimators.rtmpose import RTMPoseEstimator

        return RTMPoseEstimator(device=device)
    if name == "hrnet":
        from football_orient_pose.estimators.hrnet import HRNetEstimator

        return HRNetEstimator(device=device)
    if name == "openpose":
        from football_orient_pose.estimators.openpose import OpenPoseEstimator

        return OpenPoseEstimator(device=device)
    raise ValueError(f"estimador desconhecido: {name}")


def _test_clip_dir(examples_clip: Path, test_root: Path) -> Path | None:
    """Resolve o clip do 3DSP test correspondente via info.ini (source_video = test_0000X.mp4)."""
    try:
        src = load_clip_info(examples_clip).get("source_video", "")
        test_id = Path(src).stem.split("_")[-1]  # 'test_00001' -> '00001'
        cand = test_root / test_id
        return cand if (cand / "gt" / "gt.txt").exists() else None
    except Exception:  # noqa: BLE001
        return None


def process_clip(
    clip_dir: Path, detector, pose, test_root: Path, out_dir: Path, crops_root: Path
) -> int:
    clip_id = clip_dir.name
    test_clip = _test_clip_dir(clip_dir, test_root)
    finisher = load_finisher_boxes(test_clip) if test_clip else {}
    if not finisher:
        print(f"  [{clip_id}] sem caixa de referência do 3DSP → heurística (maior área)")

    viz_dir = out_dir / clip_id
    viz_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted((clip_dir / "img").glob("*.jpg"))
    # passada 1: detecta tudo e rastreia o finalizador ao longo do clip
    frames = [cv2.imread(str(fp)) for fp in frame_paths]
    boxes_per_frame = [detections_to_arrays(detector.detect(f))[0] for f in frames]
    ref_boxes = [finisher.get(int(fp.stem)) for fp in frame_paths]
    picks = track_finisher(boxes_per_frame, ref_boxes)

    # passada 2: crop + pose + viz no finalizador rastreado
    crops, params, bboxes = [], [], []
    for fp, frame, boxes, pick in zip(frame_paths, frames, boxes_per_frame, picks):
        idx = int(fp.stem)
        if pick is None:
            print(f"  [{clip_id}] frame {idx:03d}: sem detecção (pulado)")
            continue
        res = crop_and_pose(frame, boxes[pick], pose, all_boxes=boxes)

        vis = draw_boxes(frame, res.all_boxes, color=(160, 160, 160), thickness=1)
        vis = draw_boxes(vis, res.finisher_box.reshape(1, 4), color=(0, 0, 255), thickness=2)
        vis = draw_skeleton(vis, res.keypoints_frame)
        cv2.imwrite(str(viz_dir / f"frame_{idx:03d}.png"), vis)
        crop_vis = draw_skeleton(res.crop, res.keypoints_crop)
        cv2.imwrite(str(viz_dir / f"crop_{idx:03d}.png"), crop_vis)

        crops.append(res.crop)
        params.append(res.crop_params)
        bboxes.append(res.finisher_box)

    if crops:
        write_crop_clip(
            crops_root, clip_id, crops, params, bboxes,
            meta={"fonte": "examples", "source_clip": str(clip_dir),
                  "crop_mode": "tight", "detector": detector.name, "pose": pose.name},
        )
    print(f"  [{clip_id}] {len(crops)} frames → {viz_dir} + {crops_root / clip_id}")
    return len(crops)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline ponta-a-ponta nos examples (#126)")
    p.add_argument("--pose", default="rtmpose", choices=["rtmpose", "hrnet", "openpose"])
    p.add_argument("--data-root", type=Path, default=Path("data/clips/examples"))
    p.add_argument("--test-root", type=Path, default=Path("data/test"))
    p.add_argument("--out", type=Path, default=Path("results/pipeline"))
    p.add_argument("--crops-root", type=Path, default=Path("data/crops/examples"))
    p.add_argument("--clips", nargs="*", default=None, help="ids específicos (default: todos)")
    p.add_argument("--weights", default="yolo26x.pt", help="peso do YOLO (vencedor: yolo26x.pt)")
    p.add_argument("--device", default="cuda", help="cpu|cuda")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    detector = YOLO26Detector(weights=args.weights, device=args.device)
    pose = _build_pose(args.pose, args.device)
    print(f"Detector: {detector.name} ({args.weights}) | Pose: {pose.name} | device: {args.device}")

    clips = sorted(p for p in args.data_root.iterdir() if (p / "img").is_dir())
    if args.clips:
        clips = [c for c in clips if c.name in set(args.clips)]

    total = 0
    for clip_dir in clips:
        total += process_clip(clip_dir, detector, pose, args.test_root, args.out, args.crops_root)
    print(f"\nConcluído: {len(clips)} clip(s), {total} frames processados.")


if __name__ == "__main__":
    main()
