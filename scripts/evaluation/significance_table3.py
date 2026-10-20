#!/usr/bin/env python
"""Significância estatística da matriz 2×2 (Tabela III) sobre as 10 execuções por cenário.

Lê ``docs/experiments/table3_runs.csv`` (uma linha por execução × cenário × split) e, para cada
métrica da validação, emite:
- **t de Welch** e **Mann-Whitney U exato** para as quatro comparações do texto
  (C−A, D−B, B−A, D−C), com correção de **Holm** sobre as quatro;
- a **interação** (D − C − B + A) como contraste com graus de liberdade de Welch-Satterthwaite,
  com IC 95% pelo t.

Sem ANOVA clássica: ela supõe variâncias iguais entre as células, e aqui elas diferem até 41×
(PCK@0,2: B contra D-FULL). Num 2×2 a ANOVA testa os mesmos contrastes; a versão de Welch é a
robusta. O Mann-Whitney entra porque B e C não são normais (cada um tem uma execução atípica).

Os testes cobrem a variação entre sementes de treino, sempre na mesma divisão de validação;
não dizem nada sobre generalização a outros dados.

Uso:
    python scripts/evaluation/significance_table3.py [caminho/para/table3_runs.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

DEFAULT_CSV = Path(__file__).parents[2] / "docs" / "experiments" / "table3_runs.csv"
CELLS = ["A", "B", "C", "D-FULL"]  # A scratch, B scratch+aug, C TL, D-FULL TL+aug
METRICS = ["pck_02", "pdj_05", "oks", "oks_coco", "oks_coco_map", "mpjpe_2d_px"]
# (minuendo, subtraendo, o que mede)
CONTRASTS = [
    ("C", "A", "TL sem aug"),
    ("D-FULL", "B", "TL com aug"),
    ("B", "A", "aug do zero"),
    ("D-FULL", "C", "aug com TL"),
]
INTERACTION = {"D-FULL": 1.0, "C": -1.0, "B": -1.0, "A": 1.0}


def load(path: Path) -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                # aceita também o execucoes.csv antigo (coluna cenario_artigo)
                cells.setdefault(row.get("scenario") or row["cenario_artigo"], []).append(row)
    return cells


def welch_contrast(
    groups: dict[str, np.ndarray], weights: dict[str, float]
) -> tuple[float, float, float, float]:
    """Contraste linear Σ wᵢ·médiaᵢ com variâncias separadas (Welch-Satterthwaite).

    Retorna (estimativa, meia-largura do IC 95%, graus de liberdade, p bilateral).
    """
    terms = [(w, groups[k]) for k, w in weights.items()]
    est = sum(w * x.mean() for w, x in terms)
    parts = [w**2 * x.var(ddof=1) / len(x) for w, x in terms]
    se = float(np.sqrt(sum(parts)))
    df = sum(parts) ** 2 / sum(p**2 / (len(x) - 1) for p, (_, x) in zip(parts, terms))
    half = float(stats.t.ppf(0.975, df) * se)
    p = float(2 * stats.t.sf(abs(est) / se, df))
    return float(est), half, float(df), p


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV,
                    help="CSV das execuções (padrão: docs/experiments/table3_runs.csv)")
    args = ap.parse_args()

    cells = load(args.csv)
    for metric in METRICS:
        v = {k: np.array([float(r[metric]) for r in cells[k]]) for k in CELLS}
        n = len(v["A"])
        print(f"\n== {metric} (n = {n} por cenário) ==")
        for k in CELLS:
            x = v[k]
            print(f"  {k:7s} {x.mean():7.2f} ± {x.std(ddof=1):.2f}  [{x.min():.2f}–{x.max():.2f}]")
        variances = [x.var(ddof=1) for x in v.values()]
        print(f"  razão entre a maior e a menor variância: {max(variances) / min(variances):.1f}")

        welch, mwu, lines = [], [], []
        for hi, lo, label in CONTRASTS:
            delta, half, _, p = welch_contrast(v, {hi: 1.0, lo: -1.0})
            welch.append(p)
            mwu.append(stats.mannwhitneyu(v[hi], v[lo], method="exact").pvalue)
            lines.append(f"  {hi}−{lo} ({label}): Δ = {delta:+.2f} ± {half:.2f} (IC 95%)")
        for line, pw, pm in zip(lines, holm(welch), holm(mwu)):
            print(f"{line}  Welch p_Holm = {pw:.1e}  Mann-Whitney exato p_Holm = {pm:.1e}")

        inter, half, df, p = welch_contrast(v, INTERACTION)
        print(f"  interação (D − C − B + A) = {inter:+.2f} ± {half:.2f} (IC 95%, t de Welch, "
              f"gl = {df:.1f})  p = {p:.1e}")


if __name__ == "__main__":
    main()
