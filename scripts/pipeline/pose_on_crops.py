#!/usr/bin/env python
"""Inferência de pose em crops PRONTOS — ex.: os crops frouxos do 3DSP test (Épico #126).

Roda um estimador de pose em cada ``<crops-root>/<clip>/img/NNN.jpg`` e salva o esqueleto desenhado.
Diferente do ``demo_examples`` (que detecta + recorta), aqui é **só inferência** num crop existente.

Motivo: o modelo **fine-tunado** do Épico 2 foi treinado nos crops frouxos do 3DSP, então é nesses
crops (sua distribuição de treino) que ele produz pose coerente — sem o problema do crop justo.

Uso:
    python scripts/pipeline/pose_on_crops.py --crops-root data/3dsp/test \\
        --clips 00001 00004 00006 --pose finetuned \\
        --checkpoint results/runs/.../cenario_D-OCCL/best_PCK.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from demo_examples import _build_pose  # reusa o builder (rtmpose/finetuned/hrnet/openpose)

from football_orient_pose.utils.viz import draw_skeleton


def _resolve_root(given: Path) -> Path:
    for cand in (given, Path("data/3dsp/test"), Path("data/test")):
        if cand.exists():
            return cand
    return given


def process_clip(clip_dir: Path, pose, out_dir: Path, upscale: int = 3) -> int:
    viz_dir = out_dir / clip_dir.name
    viz_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in sorted((clip_dir / "img").glob("*.jpg")):
        crop = cv2.imread(str(fp))
        if crop is None:
            continue
        kps = pose.predict_h3wb(crop)  # (17, 2) no espaço do crop
        # amplia pra visualização (nearest) e escala os keypoints junto
        big = cv2.resize(crop, (crop.shape[1] * upscale, crop.shape[0] * upscale),
                         interpolation=cv2.INTER_NEAREST)
        vis = draw_skeleton(big, kps * upscale)
        cv2.imwrite(str(viz_dir / f"pose_{int(fp.stem):03d}.png"), vis)
        n += 1
    print(f"  [{clip_dir.name}] {n} crops → {viz_dir}")
    return n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inferência de pose em crops prontos (3DSP)")
    p.add_argument("--crops-root", type=Path, default=Path("data/3dsp/test"))
    p.add_argument("--clips", nargs="*", default=None, help="ids dos clips (default: todos)")
    p.add_argument("--pose", default="finetuned",
                   choices=["rtmpose", "finetuned", "hrnet", "openpose"])
    p.add_argument("--checkpoint", default=None, help="--pose finetuned: .pth do Épico 2")
    p.add_argument("--config", default=None, help="--pose finetuned: config MMPose (auto)")
    p.add_argument("--out", type=Path, default=Path("results/pose_crops"))
    p.add_argument("--device", default="cuda", help="cpu|cuda")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = _resolve_root(args.crops_root)
    pose = _build_pose(args.pose, args.device, checkpoint=args.checkpoint, config=args.config)
    print(f"Pose: {pose.name} | crops-root: {root} | device: {args.device}")

    clips = sorted(p for p in root.iterdir() if (p / "img").is_dir())
    if args.clips:
        clips = [c for c in clips if c.name in set(args.clips)]

    total = sum(process_clip(c, pose, args.out) for c in clips)
    print(f"\nConcluído: {len(clips)} clip(s), {total} crops.")


if __name__ == "__main__":
    main()
