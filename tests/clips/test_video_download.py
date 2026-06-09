from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from football_orient_pose.video_download import download_video


def _fake_ytdlp(captured: dict, tmp_path: Path) -> types.ModuleType:
    class FakeYDL:
        def __init__(self, options):
            captured["options"] = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            captured["url"] = url
            captured["download"] = download
            return {"title": "Jogo do Brasil", "ext": "mp4"}

        def prepare_filename(self, info):
            return str(tmp_path / "Jogo do Brasil.mp4")

    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = FakeYDL
    return module


def test_download_video_requires_ytdlp(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp", None)  # força ImportError no import
    with pytest.raises(ImportError, match="yt-dlp"):
        download_video("http://x", output_dir=tmp_path, name="x")


def test_download_video_with_name_returns_deterministic_path(monkeypatch, tmp_path) -> None:
    captured: dict = {}
    monkeypatch.setitem(sys.modules, "yt_dlp", _fake_ytdlp(captured, tmp_path))

    out = download_video("http://x", output_dir=tmp_path, name="brasil_x_y", max_height=720)

    assert out == tmp_path / "brasil_x_y.mp4"
    assert captured["download"] is True
    assert "height<=720" in captured["options"]["format"]
    assert captured["options"]["noplaylist"] is True


def test_download_video_without_name_uses_prepared_filename(monkeypatch, tmp_path) -> None:
    captured: dict = {}
    monkeypatch.setitem(sys.modules, "yt_dlp", _fake_ytdlp(captured, tmp_path))

    out = download_video("http://x", output_dir=tmp_path)

    assert out == tmp_path / "Jogo do Brasil.mp4"
