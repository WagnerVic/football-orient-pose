#!/usr/bin/env python
"""Gera GIFs animados a partir das PNGs do showcase (Épico #126).

Pós-processamento puro (sem GPU, sem re-rodar o pipeline): para cada clip em `results/showcase/`,
agrupa as PNGs por prefixo (`frame_`, `crop_`) e monta um GIF por grupo. Leve e embutível no doc.

- `all_players/<clip>/frame_*.png`            → `gifs/all_players/<clip>.gif`        (1 por clip)
- `finisher/<clip>/{frame_,crop_}*.png`       → `gifs/finisher/<clip>_frame.gif` + `<clip>_crop.gif`

Uso:
    python scripts/pipeline/make_gifs.py                 # ambos os showcases
    python scripts/pipeline/make_gifs.py --src finisher --clips example_01
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from football_orient_pose.utils.viz import frames_to_gif

_FRAME_RE = re.compile(r"^(.*)_(\d+)\.png$")


def _group_pngs(clip_dir: Path) -> dict[str, list[Path]]:
    """Agrupa as PNGs do clip por prefixo (frame/crop), ordenadas pelo número."""
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for fp in clip_dir.glob("*.png"):
        m = _FRAME_RE.match(fp.name)
        if m:
            groups[m.group(1)].append((int(m.group(2)), fp))
    return {prefix: [p for _, p in sorted(items)] for prefix, items in groups.items()}


def process_clip(
    clip_dir: Path, out_dir: Path, fps: float, width: int, crop_width: int,
) -> int:
    groups = _group_pngs(clip_dir)
    if not groups:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for prefix, paths in groups.items():
        frames = [img for fp in paths if (img := cv2.imread(str(fp))) is not None]
        if not frames:
            continue
        # grupo único → <clip>.gif; múltiplos → <clip>_<prefixo>.gif
        stem = clip_dir.name if len(groups) == 1 else f"{clip_dir.name}_{prefix}"
        target_width = crop_width if prefix == "crop" else width
        out = frames_to_gif(frames, out_dir / f"{stem}.gif", fps=fps, target_width=target_width)
        print(f"  {out}  ({len(frames)} quadros, {target_width}px)")
        n += 1
    return n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GIFs animados do showcase (#126)")
    p.add_argument("--src", default="all", choices=["all", "finisher", "all_players"],
                   help="qual showcase processar (default: ambos)")
    p.add_argument("--showcase-root", type=Path, default=Path("results/showcase"))
    p.add_argument("--out", type=Path, default=Path("results/showcase/gifs"))
    p.add_argument("--clips", nargs="*", default=None, help="ids específicos (default: todos)")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--width", type=int, default=640, help="largura dos GIFs de frame (downscale)")
    p.add_argument("--crop-width", type=int, default=300, help="largura dos GIFs de crop (upscale)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    showcases = ["finisher", "all_players"] if args.src == "all" else [args.src]

    total = 0
    for name in showcases:
        src = args.showcase_root / name
        if not src.is_dir():
            print(f"[{name}] ausente em {src} — pulado")
            continue
        clips = sorted(d for d in src.iterdir() if d.is_dir())
        if args.clips:
            clips = [c for c in clips if c.name in set(args.clips)]
        print(f"[{name}] {len(clips)} clip(s):")
        for clip_dir in clips:
            total += process_clip(
                clip_dir, args.out / name, args.fps, args.width, args.crop_width,
            )
    print(f"\nConcluído: {total} GIF(s) em {args.out}")


if __name__ == "__main__":
    main()
