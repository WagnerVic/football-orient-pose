#!/usr/bin/env python
"""Showcase "pose em TODOS os jogadores" — replica o baseline Reis et al. (foco Brasil, Épico #126).

Para cada frame, o YOLO26x detecta todos os jogadores, recorta cada caixa, estima a pose e re-cola
todos os esqueletos no frame (mesma ideia da Fig. 5 do Reis). Não há finalizador/seleção: poseia
todo mundo. É o showcase **100% automático** do Brasil, que não tem anotação de finalizador.

Saída: 1 PNG compositado por frame em ``results/showcase/all_players/<clip>/frame_NNN.png``
(results/ é gitignored; nada de crop por jogador é salvo — são intermediários em memória).

Uso:
    python scripts/pipeline/pose_all_players.py --data-root data/clips/brazil --pose rtmpose
    python scripts/pipeline/pose_all_players.py --data-root data/clips/examples --clips example_01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from demo_examples import _build_pose  # reusa o builder (rtmpose/finetuned/hrnet/openpose)

from football_orient_pose.detection import YOLO26Detector, detections_to_arrays
from football_orient_pose.pipeline import pose_all
from football_orient_pose.utils.viz import draw_boxes, draw_skeleton


def process_clip(
    clip_dir: Path, detector, pose, out_dir: Path, min_box_height: float,
) -> int:
    clip_id = clip_dir.name
    viz_dir = out_dir / clip_id
    viz_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for fp in sorted((clip_dir / "img").glob("*.jpg")):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        boxes, _scores = detections_to_arrays(detector.detect(frame))
        results = pose_all(frame, boxes, pose, min_box_height=min_box_height)

        vis = draw_boxes(frame, boxes, color=(160, 160, 160), thickness=1)
        for res in results:  # estilo único, sem destacar ninguém (fiel ao Reis)
            vis = draw_skeleton(vis, res.keypoints_frame)
        cv2.imwrite(str(viz_dir / f"frame_{int(fp.stem):03d}.png"), vis)
        n += 1
    print(f"  [{clip_id}] {n} frames → {viz_dir}")
    return n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Showcase pose em todos os jogadores (Reis, #126)")
    p.add_argument("--data-root", type=Path, default=Path("data/clips/brazil"))
    p.add_argument("--clips", nargs="*", default=None, help="ids específicos (default: todos)")
    p.add_argument("--pose", default="rtmpose",
                   choices=["rtmpose", "finetuned", "hrnet", "openpose"])
    p.add_argument("--checkpoint", default=None, help="--pose finetuned: .pth do Épico 2")
    p.add_argument("--config", default=None, help="--pose finetuned: config MMPose (auto)")
    p.add_argument("--weights", default="yolo26x.pt", help="peso do YOLO (vencedor: yolo26x.pt)")
    p.add_argument("--out", type=Path, default=Path("results/showcase/all_players"))
    p.add_argument("--min-box-height", type=float, default=40.0,
                   help="descarta caixas mais baixas que isto (px); jogador distante = crop ruim")
    p.add_argument("--device", default="cuda", help="cpu|cuda")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    detector = YOLO26Detector(weights=args.weights, device=args.device)
    pose = _build_pose(args.pose, args.device, checkpoint=args.checkpoint, config=args.config)
    print(f"Detector: {detector.name} ({args.weights}) | Pose: {pose.name} | device: {args.device}")

    clips = sorted(p for p in args.data_root.iterdir() if (p / "img").is_dir())
    if args.clips:
        clips = [c for c in clips if c.name in set(args.clips)]

    total = sum(process_clip(c, detector, pose, args.out, args.min_box_height) for c in clips)
    print(f"\nConcluído: {len(clips)} clip(s), {total} frames → {args.out}")


if __name__ == "__main__":
    main()
