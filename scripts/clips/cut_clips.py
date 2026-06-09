#!/usr/bin/env python
"""Corta um vídeo bruto em clips estruturados (estilo 3DSP) a partir de intervalos.

Recebe um vídeo e um JSON de intervalos e gera, para cada um, um clip em
``data/clips/<id>/`` com ``img/{001..0NN}.jpg`` + ``info.ini``.

Uso:
    python scripts/cut_clips.py \\
        --video data/raw/brasil.mp4 \\
        --intervals intervals_brazil.json \\
        [--n 20] \\
        [--root data/clips] \\
        [--fps 25] [--step-ms 40]

Formato do intervals.json:
    [
      {"id": "brazil_01", "start_ms": 123000, "end_ms": 124000,
       "label": "Finalização", "game": "Brasil x Y (2024)"}
    ]

Saída:
    - data/clips/<id>/img/001.jpg ... + info.ini, por intervalo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from football_orient_pose.clip_extractor import cut_clip, write_clip


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Corta vídeo bruto em clips estruturados (3DSP)")
    p.add_argument("--video", required=True, type=Path, help="Vídeo de origem (.mp4)")
    p.add_argument(
        "--intervals", required=True, type=Path,
        help="JSON com [{id, start_ms, end_ms, label?, game?}]",
    )
    p.add_argument("--n", type=int, default=20, help="Frames por clip (default 20)")
    p.add_argument("--root", type=Path, default=Path("data/clips"), help="Raiz de saída")
    p.add_argument("--fps", type=int, default=25, help="FPS registrado no info.ini")
    p.add_argument("--step-ms", type=int, default=40, help="Passo (ms) registrado no info.ini")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    intervals = json.loads(args.intervals.read_text(encoding="utf-8"))

    for it in intervals:
        clip_id = it["id"]
        frames = cut_clip(args.video, it["start_ms"], it["end_ms"], n=args.n)
        meta = {
            "source_video": str(args.video),
            "game": it.get("game", ""),
            "label": it.get("label", "Finalização"),
            "start_ms": it["start_ms"],
            "end_ms": it["end_ms"],
            "fps": args.fps,
            "step_ms": args.step_ms,
            "notes": it.get("notes", ""),
        }
        clip_dir = write_clip(clip_id, frames, meta, root=args.root)
        print(f"  {clip_id}: {len(frames)} frames -> {clip_dir}")

    print(f"OK — {len(intervals)} clip(s) gerado(s) em {args.root}")


if __name__ == "__main__":
    main()
