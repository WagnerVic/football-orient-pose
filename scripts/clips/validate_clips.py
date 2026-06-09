#!/usr/bin/env python
"""Valida clips estruturados contra as regras do formato (ver docs/vision/formato-clips.md).

Para cada clip em ``--root``, checa: nº de frames, resolução mínima, campos
obrigatórios do ``info.ini`` e que os loaders existentes (``load_clip_image`` /
``load_clip_info``) carregam sem erro. Falha (exit code 1) se houver qualquer
erro ou se houver menos que ``--min-clips`` clips.

Uso:
    python scripts/validate_clips.py [--root data/clips] [--n 20] [--min-clips 5] [--min-height 720]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from football_orient_pose.utils.data_io import load_clip_image, load_clip_info

REQUIRED_INFO_FIELDS = {
    "id", "source_video", "game", "label",
    "start_ms", "end_ms", "fps", "step_ms", "num_frames",
}


def validate_clip(clip_dir: Path, n: int = 20, min_height: int = 720) -> list[str]:
    """Valida um clip e devolve a lista de erros (vazia = válido)."""
    errors: list[str] = []
    name = clip_dir.name

    img_dir = clip_dir / "img"
    imgs = sorted(img_dir.glob("*.jpg")) if img_dir.exists() else []
    if len(imgs) != n:
        errors.append(f"{name}: {len(imgs)} frames em img/ (esperado {n})")

    try:
        img = load_clip_image(clip_dir, 1)
        if img.shape[0] < min_height:
            errors.append(f"{name}: altura {img.shape[0]}px < {min_height}px")
    except Exception as exc:  # noqa: BLE001 — reporta qualquer falha de leitura
        errors.append(f"{name}: load_clip_image falhou ({exc})")

    if not (clip_dir / "info.ini").exists():
        errors.append(f"{name}: info.ini ausente")
    else:
        try:
            info = load_clip_info(clip_dir)
            missing = REQUIRED_INFO_FIELDS - set(info)
            if missing:
                errors.append(f"{name}: info.ini sem campos {sorted(missing)}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: load_clip_info falhou ({exc})")

    return errors


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Valida clips estruturados (formato 3DSP)")
    p.add_argument("--root", type=Path, default=Path("data/clips"))
    p.add_argument("--n", type=int, default=20, help="Frames esperados por clip")
    p.add_argument("--min-clips", type=int, default=5)
    p.add_argument("--min-height", type=int, default=720)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.root.exists():
        print(f"ERRO: diretório não encontrado: {args.root}")
        return 1

    # um clip é qualquer diretório que contenha uma subpasta img/ — acha tanto
    # clips planos (data/clips/<id>) quanto aninhados por fonte (data/clips/brazil/<id>)
    clips = sorted({p.parent for p in args.root.rglob("img") if p.is_dir()})
    errors: list[str] = []
    for clip in clips:
        errors.extend(validate_clip(clip, n=args.n, min_height=args.min_height))

    if len(clips) < args.min_clips:
        errors.append(f"apenas {len(clips)} clip(s) em {args.root} (< {args.min_clips})")

    if errors:
        print(f"❌ {len(errors)} problema(s) em {len(clips)} clip(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"✅ {len(clips)} clip(s) válido(s) em {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
