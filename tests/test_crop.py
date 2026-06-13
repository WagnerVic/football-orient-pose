from __future__ import annotations

import numpy as np
import pytest

from football_orient_pose.crop import (
    CropParams,
    crop_to_frame,
    frame_to_crop,
    make_crop,
    project_between_crops,
)


def _frame(h: int = 720, w: int = 1280) -> np.ndarray:
    # gradiente para detectar stretch/distorção (conteúdo não-uniforme)
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    f[:, :, 1] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    return f


def test_make_crop_shapes_and_dtype() -> None:
    frame = _frame()
    box = np.array([400, 300, 440, 384], dtype=np.float32)  # jogador ~40×84
    for mode in ("tight", "loose"):
        crop, p = make_crop(frame, box, mode=mode)
        assert crop.shape == (100, 100, 3) and crop.dtype == np.uint8
        assert isinstance(p, CropParams) and p.size == 100


def test_size_param_is_respected() -> None:
    crop, p = make_crop(_frame(), (400, 300, 440, 384), mode="tight", size=256)
    assert crop.shape == (256, 256, 3) and p.size == 256


def test_letterbox_preserves_aspect_no_stretch() -> None:
    # caixa alta (40×80, razão 1:2) → após letterbox, altura=100 e largura<100, com padding lateral
    crop, p = make_crop(_frame(), (400, 300, 440, 380), mode="tight", margin=0.0)
    new_h = round((380 - 300) * p.scale)
    new_w = round((440 - 400) * p.scale)
    assert new_h == 100  # lado maior preenche
    assert new_w < 100 and p.pad_left > 0 and p.pad_top == 0  # padding na largura
    # colunas das bordas (área de padding) ficam pretas
    assert crop[:, 0].sum() == 0 and crop[:, -1].sum() == 0


def test_loose_region_is_larger_than_tight() -> None:
    frame, box = _frame(), np.array([600, 300, 640, 384], dtype=np.float32)
    _, pt = make_crop(frame, box, mode="tight")
    _, pl = make_crop(frame, box, mode="loose")
    # área da região no frame = (size/scale)^2 (lado maior); loose deve recortar mais "fundo"
    assert (1 / pl.scale) > (1 / pt.scale)


def test_roundtrip_frame_crop_under_1px() -> None:
    _, p = make_crop(_frame(), (400, 300, 460, 400), mode="tight")
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, size=(50, 2))
    back = frame_to_crop(crop_to_frame(pts, p), p)
    assert np.max(np.abs(back - pts)) < 1e-6  # transform é linear → erro ~zero


def test_project_between_crops_consistent() -> None:
    frame, box = _frame(), np.array([500, 250, 560, 390], dtype=np.float32)
    _, p_tight = make_crop(frame, box, mode="tight")
    _, p_loose = make_crop(frame, box, mode="loose")
    pt = np.array([50.0, 50.0])
    in_loose = project_between_crops(pt, p_tight, p_loose)
    back = project_between_crops(in_loose, p_loose, p_tight)
    np.testing.assert_allclose(back, pt, atol=1e-6)


def test_cropparams_json_roundtrip() -> None:
    p = CropParams(x0=10.0, y0=20.0, scale=1.25, pad_top=0.0, pad_left=8.0, size=100)
    assert CropParams.from_json(p.to_json()) == p


def test_box_partly_outside_frame_still_valid() -> None:
    # caixa estourando a borda esquerda/superior
    crop, p = make_crop(_frame(), (-30, -20, 50, 120), mode="tight")
    assert crop.shape == (100, 100, 3)
    assert p.x0 == 0 and p.y0 == 0  # clampado
    # round-trip continua consistente
    pts = np.array([[10.0, 10.0], [90.0, 90.0]])
    np.testing.assert_allclose(frame_to_crop(crop_to_frame(pts, p), p), pts, atol=1e-6)


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode inválido"):
        make_crop(_frame(), (0, 0, 10, 10), mode="medium")
