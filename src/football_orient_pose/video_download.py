"""Download de vídeo (YouTube etc.) para a pasta de fontes brutas.

Usado para obter o vídeo do jogo do Brasil (Épico 13, #143) antes do corte em
clips por ``scripts/clips/cut_clips.py``. Requer a dependência opcional
``yt-dlp`` (``pip install -e '.[download]'`` ou ``pip install yt-dlp``).
"""

from __future__ import annotations

from pathlib import Path


def download_video(
    url: str,
    output_dir: str | Path = "data/raw/videos",
    name: str | None = None,
    max_height: int = 720,
) -> Path:
    """Baixa um vídeo e retorna o caminho do arquivo salvo.

    Parameters
    ----------
    url : str
        URL do vídeo (ex.: link do YouTube).
    output_dir : str | Path
        Pasta de destino. Default ``data/raw/videos`` (ignorada pelo git).
    name : str | None
        Nome do arquivo **sem extensão** (ex.: ``"brasil_x_y"`` → ``brasil_x_y.mp4``).
        Se ``None``, usa o título do vídeo.
    max_height : int
        Altura máxima do stream (default 720p — alvo do broadcast).

    Returns
    -------
    Path
        Caminho do arquivo de vídeo baixado.

    Raises
    ------
    ImportError
        Se ``yt-dlp`` não estiver instalado.
    """
    try:
        import yt_dlp
    except ImportError as exc:
        raise ImportError(
            "download_video requer a dependência opcional 'yt-dlp'. "
            "Instale com \"pip install -e '.[download]'\" ou \"pip install yt-dlp\"."
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outtmpl = str(out_dir / (f"{name}.%(ext)s" if name else "%(title)s.%(ext)s"))
    options = {
        # stream progressivo (sem precisar de ffmpeg p/ merge), até max_height
        "format": (
            f"best[ext=mp4][height<={max_height}]"
            f"/best[height<={max_height}]/best"
        ),
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        if name:
            ext = info.get("ext", "mp4")
            return out_dir / f"{name}.{ext}"
        return Path(ydl.prepare_filename(info))
