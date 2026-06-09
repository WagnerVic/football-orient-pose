#!/usr/bin/env python
"""Valida clips estruturados contra as regras do formato (ver docs/vision/formato-clips.md).

Para cada clip em ``--root``, checa **consistência interna** (não um nº fixo de
frames): a quantidade de imagens bate com o ``num_frames`` do próprio ``info.ini``,
os arquivos são contíguos ``001..N`` (necessário para os loaders), a resolução é
mínima, os campos do ``info.ini`` existem e ``load_clip_image``/``load_clip_info``
carregam sem erro. Assim, cada fonte pode ter um nº de frames diferente
(ex.: examples=20, brazil=10) e um ``validate --root data/clips`` global funciona.

Falha (exit code 1) se houver qualquer erro ou menos que ``--min-clips`` clips.

Uso:
    python scripts/clips/validate_clips.py [--root data/clips] [--min-clips 5]
        [--min-height 720] [--min-frames 1]
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


def validate_clip(clip_dir: Path, min_height: int = 720, min_frames: int = 1) -> list[str]:
    """Valida um clip pela consistência com o próprio info.ini. Devolve erros (vazia = válido)."""
    errors: list[str] = []
    name = clip_dir.name

    # info.ini é a fonte da verdade — sem ele não dá para checar consistência
    if not (clip_dir / "info.ini").exists():
        return [f"{name}: info.ini ausente"]
    try:
        info = load_clip_info(clip_dir)
    except Exception as exc:  # noqa: BLE001
        return [f"{name}: load_clip_info falhou ({exc})"]

    missing = REQUIRED_INFO_FIELDS - set(info)
    if missing:
        errors.append(f"{name}: info.ini sem campos {sorted(missing)}")

    try:
        declared = int(info["num_frames"])
    except (KeyError, ValueError):
        errors.append(f"{name}: num_frames inválido/ausente no info.ini")
        return errors

    if declared < min_frames:
        errors.append(f"{name}: num_frames {declared} < mínimo {min_frames}")

    # img/ deve bater com num_frames E ser contíguo 001..declared (loaders leem em sequência)
    img_dir = clip_dir / "img"
    imgs = sorted(img_dir.glob("*.jpg")) if img_dir.exists() else []
    if len(imgs) != declared:
        errors.append(f"{name}: {len(imgs)} frames em img/ ≠ num_frames={declared} (info.ini)")
    expected = {f"{i:03d}" for i in range(1, declared + 1)}
    actual = {p.stem for p in imgs}
    if actual != expected:
        faltando = sorted(expected - actual)[:5]
        extras = sorted(actual - expected)[:5]
        errors.append(
            f"{name}: img não contígua 001..{declared:03d} "
            f"(faltando={faltando} extras={extras})"
        )

    try:
        img = load_clip_image(clip_dir, 1)
        if img.shape[0] < min_height:
            errors.append(f"{name}: altura {img.shape[0]}px < {min_height}px")
    except Exception as exc:  # noqa: BLE001 — reporta qualquer falha de leitura
        errors.append(f"{name}: load_clip_image falhou ({exc})")

    return errors


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Valida clips estruturados (formato 3DSP)")
    p.add_argument("--root", type=Path, default=Path("data/clips"))
    p.add_argument("--min-clips", type=int, default=5)
    p.add_argument("--min-height", type=int, default=720)
    p.add_argument("--min-frames", type=int, default=1, help="Piso de frames por clip")
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
        errors.extend(validate_clip(clip, min_height=args.min_height, min_frames=args.min_frames))

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
