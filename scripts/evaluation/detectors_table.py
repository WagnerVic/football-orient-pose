#!/usr/bin/env python
"""Agrega os JSONs por detector em uma tabela comparativa (markdown + LaTeX). Épico #113, task #116.

Lê todos os ``results/tables/detector_*.json`` (gerados por ``eval_detectors.py``) e emite:
- uma tabela **markdown** no terminal (relatório #118), com o melhor por coluna em **negrito**;
- um arquivo **LaTeX** em ``results/tables/detectores.tex`` (para o artigo).

Colunas: mAP, AP50, AP75, AR_small/medium/large (recall por tamanho) e F1.

Uso:
    python scripts/evaluation/detectors_table.py [--dir results/tables]
        [--tex results/tables/detectores.tex]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (chave no JSON, rótulo na tabela)
COLUMNS = [
    ("mAP", "mAP"),
    ("AP50", "AP50"),
    ("AP75", "AP75"),
    ("AR_small", "AR_s"),
    ("AR_medium", "AR_m"),
    ("AR_large", "AR_l"),
    ("f1", "F1"),
]


def load_results(dir_path: str | Path) -> list[dict]:
    dir_path = Path(dir_path)
    files = sorted(dir_path.glob("detector_*.json"))
    if not files:
        raise FileNotFoundError(f"nenhum detector_*.json em {dir_path}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _best_per_column(results: list[dict]) -> dict[str, float]:
    # ignora -1 (COCOeval: "sem objetos desse tamanho") ao escolher o melhor
    best = {}
    for key, _ in COLUMNS:
        vals = [r.get(key, 0.0) for r in results if r.get(key, 0.0) >= 0]
        best[key] = max(vals) if vals else -1.0
    return best


def _fmt_cell(value: float, best: float, *, bold: str, end: str = "") -> str:
    if value < 0:  # COCOeval N/A (ex.: AR_large sem objetos grandes)
        return "—"
    cell = f"{value * 100:.1f}"
    if best >= 0 and abs(value - best) < 1e-9:
        return f"{bold}{cell}{end}"
    return cell


def to_markdown(results: list[dict]) -> str:
    best = _best_per_column(results)
    header = "| Detector | " + " | ".join(lbl for _, lbl in COLUMNS) + " |"
    sep = "|" + "---|" * (len(COLUMNS) + 1)
    lines = [header, sep]
    for r in sorted(results, key=lambda x: x.get("mAP", 0.0), reverse=True):
        cells = [_fmt_cell(r.get(key, 0.0), best[key], bold="**", end="**") for key, _ in COLUMNS]
        lines.append(f"| {r['detector']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_latex(results: list[dict]) -> str:
    best = _best_per_column(results)
    col_fmt = "l" + "r" * len(COLUMNS)
    head = " & ".join(["Detector"] + [lbl for _, lbl in COLUMNS]) + r" \\"
    rows = []
    for r in sorted(results, key=lambda x: x.get("mAP", 0.0), reverse=True):
        cells = [_fmt_cell(r.get(key, 0.0), best[key], bold=r"\textbf{", end="}")
                 for key, _ in COLUMNS]
        rows.append(" & ".join([r["detector"]] + cells) + r" \\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{" + col_fmt + "}\n\\hline\n"
        + head + "\n\\hline\n" + body + "\n\\hline\n\\end{tabular}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tabela comparativa dos detectores (#116)")
    p.add_argument("--dir", type=Path, default=Path("results/tables"))
    p.add_argument("--tex", type=Path, default=Path("results/tables/detectores.tex"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results = load_results(args.dir)
    md = to_markdown(results)
    print("\nComparação dos detectores (valores em %):\n")
    print(md)
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(to_latex(results) + "\n", encoding="utf-8")
    print(f"\nLaTeX salvo em {args.tex}")


if __name__ == "__main__":
    main()
