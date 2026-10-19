#!/usr/bin/env python
"""Significância estatística da matriz 2×2 (Tabela III) sobre as 10 execuções por cenário.

Lê o ``execucoes.csv`` gerado por ``gerar_tabelas.py`` (uma linha por execução × cenário × split)
e, para cada métrica da validação, emite:
- **ANOVA de dois fatores** (inicialização × augmentation, com interação; desenho balanceado);
- **Welch t** e **Mann-Whitney U** para as quatro comparações do texto (C−A, D−B, B−A, D−C),
  com correção de **Holm** sobre as quatro;
- o termo de **interação** (D − C − B + A) com IC 95%.

O Welch e o Mann-Whitney entram porque as variâncias diferem entre células (Levene) e o C tem
uma execução atípica (seed 7) — a ANOVA clássica supõe variâncias iguais e normalidade.

Uso:
    python scripts/evaluation/significance_table3.py caminho/para/execucoes.csv
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy import stats

CELLS = ["A", "B", "C", "D-FULL"]  # A scratch, B scratch+aug, C TL, D-FULL TL+aug
METRICS = ["pck_02", "pdj_05", "oks", "oks_coco", "oks_coco_map", "mpjpe_2d_px"]
# (minuendo, subtraendo, o que mede)
CONTRASTS = [
    ("C", "A", "TL sem aug"),
    ("D-FULL", "B", "TL com aug"),
    ("B", "A", "aug do zero"),
    ("D-FULL", "C", "aug com TL"),
]


def load(path: str) -> dict[str, list[dict]]:
    cells: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "val":
                cells.setdefault(row["cenario_artigo"], []).append(row)
    return cells


def two_way_anova(v: dict[str, np.ndarray]) -> dict[str, tuple[float, float, float]]:
    """ANOVA 2×2 balanceada. Retorna {efeito: (F, p, eta² parcial)}."""
    a, b, c, d = (v[k] for k in CELLS)
    n = len(a)
    grand = np.concatenate([a, b, c, d]).mean()
    ss_tl = 2 * n * sum((m - grand) ** 2 for m in (np.r_[a, b].mean(), np.r_[c, d].mean()))
    ss_aug = 2 * n * sum((m - grand) ** 2 for m in (np.r_[a, c].mean(), np.r_[b, d].mean()))
    ss_cells = n * sum((x.mean() - grand) ** 2 for x in (a, b, c, d))
    ss_int = ss_cells - ss_tl - ss_aug
    ss_err = sum(((x - x.mean()) ** 2).sum() for x in (a, b, c, d))
    df_err = 4 * (n - 1)
    out = {}
    for name, ss in (("TL", ss_tl), ("aug", ss_aug), ("TL×aug", ss_int)):
        f_val = ss / (ss_err / df_err)
        out[name] = (f_val, stats.f.sf(f_val, 1, df_err), ss / (ss + ss_err))
    return out


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="execucoes.csv de resultados-tabela3/dados/")
    args = ap.parse_args()

    cells = load(args.csv)
    for metric in METRICS:
        v = {k: np.array([float(r[metric]) for r in cells[k]]) for k in CELLS}
        n = len(v["A"])
        print(f"\n== {metric} (n = {n} por cenário) ==")
        for k in CELLS:
            print(f"  {k:7s} {v[k].mean():7.2f} ± {v[k].std(ddof=1):.2f}  [{v[k].min():.2f}–{v[k].max():.2f}]")

        print(f"  Levene p = {stats.levene(*v.values()).pvalue:.2g}")
        for name, (f_val, p, eta) in two_way_anova(v).items():
            print(f"  ANOVA {name:7s} F(1,{4 * (n - 1)}) = {f_val:8.1f}  p = {p:.1e}  η²p = {eta:.3f}")

        welch, mwu, lines = [], [], []
        for hi, lo, label in CONTRASTS:
            x, y = v[hi], v[lo]
            se2 = x.var(ddof=1) / n + y.var(ddof=1) / n
            df = se2**2 / ((x.var(ddof=1) / n) ** 2 / (n - 1) + (y.var(ddof=1) / n) ** 2 / (n - 1))
            ci = stats.t.ppf(0.975, df) * np.sqrt(se2)
            welch.append(stats.ttest_ind(x, y, equal_var=False).pvalue)
            mwu.append(stats.mannwhitneyu(x, y).pvalue)
            lines.append(f"  {hi}−{lo} ({label}): Δ = {x.mean() - y.mean():+.2f} ± {ci:.2f} (IC 95%)")
        for line, pw, pm in zip(lines, holm(welch), holm(mwu)):
            print(f"{line}  Welch p_Holm = {pw:.1e}  Mann-Whitney p_Holm = {pm:.1e}")

        inter = v["D-FULL"].mean() - v["C"].mean() - v["B"].mean() + v["A"].mean()
        se = np.sqrt(sum(x.var(ddof=1) / n for x in v.values()))
        print(f"  interação (D − C − B + A) = {inter:+.2f} ± {1.96 * se:.2f} (IC 95%)")


if __name__ == "__main__":
    main()
