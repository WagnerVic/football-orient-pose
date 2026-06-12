from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts" / "evaluation"))

from detectors_table import load_results, to_latex, to_markdown


def _write(dir_path: Path, name: str, **vals) -> None:
    base = {"detector": name, "mAP": 0.0, "AP50": 0.0, "AP75": 0.0,
            "AR_small": 0.0, "AR_medium": 0.0, "AR_large": 0.0, "f1": 0.0}
    base.update(vals)
    (dir_path / f"detector_{name}.json").write_text(json.dumps(base), encoding="utf-8")


def test_load_and_markdown_bolds_best(tmp_path: Path) -> None:
    _write(tmp_path, "yolo26", mAP=0.80, f1=0.90)
    _write(tmp_path, "retinanet", mAP=0.60, f1=0.95)
    results = load_results(tmp_path)
    assert len(results) == 2
    md = to_markdown(results)
    # maior mAP (yolo26) aparece antes; melhor mAP e melhor F1 em negrito
    assert md.index("yolo26") < md.index("retinanet")
    assert "**80.0**" in md  # melhor mAP
    assert "**95.0**" in md  # melhor F1 (retinanet)


def test_latex_has_tabular_and_bold(tmp_path: Path) -> None:
    _write(tmp_path, "yolo26", mAP=0.80)
    tex = to_latex(load_results(tmp_path))
    assert tex.startswith("\\begin{tabular}") and "\\end{tabular}" in tex
    assert "\\textbf{80.0}" in tex


def test_cocoeval_na_minus_one_is_dash(tmp_path: Path) -> None:
    # AR_large=-1 (COCOeval: sem objetos grandes) → "—", nunca em negrito
    _write(tmp_path, "yolo26", mAP=0.80, AR_large=-1.0)
    md = to_markdown(load_results(tmp_path))
    assert "—" in md and "**—**" not in md
