"""I/O do estágio com crop (``data/crops/``) — escrita e leitura dos crops do finalizador.

Espelha o papel do ``clip_extractor`` para o estágio anterior (clips). Cada clip de crop guarda os
crops justos 100×100 (``img/``), os ``CropParams`` por frame (``crop_params/``, a chave de
reprojeção frame↔crop) e o ``info.ini``. É o insumo da anotação de keypoints (US #109): quando ela
ocorrer, cria ``posture/NNN.json`` ao lado, e o keypoint GT projeta via ``crop.py``.

Spec: ``docs/vision/formato-crops.md``.
"""

from __future__ import annotations

import configparser
import json
from pathlib import Path

import cv2
import numpy as np

from football_orient_pose.crop import CropParams


def write_crop_clip(
    root: str | Path,
    clip_id: str,
    crops: list[np.ndarray],
    params: list[CropParams],
    bboxes: list[np.ndarray],
    meta: dict | None = None,
) -> Path:
    """Escreve um clip de crops em ``root/clip_id/`` (img + crop_params + info.ini).

    ``crops``/``params``/``bboxes`` (caixa do finalizador em coord. de frame) devem ter o mesmo
    comprimento. ``meta`` vai para o ``[info]`` (ex.: ``fonte``, ``source_clip``, ``crop_mode``,
    ``detector``). Retorna o diretório do clip.
    """
    if not (len(crops) == len(params) == len(bboxes)):
        raise ValueError("crops, params e bboxes precisam ter o mesmo comprimento")

    meta = dict(meta or {})
    clip_dir = Path(root) / clip_id
    (clip_dir / "img").mkdir(parents=True, exist_ok=True)
    (clip_dir / "crop_params").mkdir(parents=True, exist_ok=True)

    for i, (crop, p, box) in enumerate(zip(crops, params, bboxes), start=1):
        cv2.imwrite(str(clip_dir / "img" / f"{i:03d}.jpg"), crop)
        record = {
            "crop_params": p.to_json(),
            "finisher_bbox_xyxy": [float(v) for v in np.asarray(box).reshape(-1)],
            "source_frame": i,
            "crop_mode": meta.get("crop_mode", "tight"),
            "detector": meta.get("detector", "yolo26x"),
        }
        (clip_dir / "crop_params" / f"{i:03d}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    cfg = configparser.ConfigParser()
    cfg["info"] = {
        "id": clip_id,
        "num_frames": str(len(crops)),
        "size": str(params[0].size if params else 100),
        **{k: str(v) for k, v in meta.items()},
    }
    with open(clip_dir / "info.ini", "w", encoding="utf-8") as f:
        cfg.write(f)
    return clip_dir


def load_crop_params(clip_dir: str | Path, frame_idx: int) -> CropParams:
    """Relê o ``CropParams`` de um frame (``crop_params/NNN.json``)."""
    path = Path(clip_dir) / "crop_params" / f"{frame_idx:03d}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    return CropParams.from_json(record["crop_params"])
