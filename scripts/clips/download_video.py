#!/usr/bin/env python
"""Baixa um vídeo (ex.: YouTube) para data/raw/videos/.

Primeiro passo da geração de clips reais (Épico 13, #143): obter o vídeo do
jogo. Depois, corte em clips com scripts/clips/cut_clips.py.

Uso:
    python scripts/clips/download_video.py \\
        --url "https://www.youtube.com/watch?v=..." \\
        [--name brasil_x_adversario] \\
        [--output-dir data/raw/videos] \\
        [--max-height 720]

Depois:
    python scripts/clips/cut_clips.py \\
        --video data/raw/videos/brasil_x_adversario.mp4 \\
        --intervals intervals_brazil.json --root data/clips/brazil

Requer a dependência opcional: pip install -e '.[download]'  (ou pip install yt-dlp)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from football_orient_pose.video_download import download_video


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baixa vídeo para data/raw/videos/")
    p.add_argument("--url", required=True, help="URL do vídeo (ex.: YouTube)")
    p.add_argument(
        "--name", default=None,
        help="Nome do arquivo sem extensão (default: título do vídeo)",
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/raw/videos"))
    p.add_argument("--max-height", type=int, default=720, help="Altura máx. do stream")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    path = download_video(
        args.url, args.output_dir, name=args.name, max_height=args.max_height
    )
    print(f"OK — vídeo salvo em {path}")
    print(f"Próximo passo: cortar em clips com scripts/clips/cut_clips.py --video {path}")


if __name__ == "__main__":
    main()
