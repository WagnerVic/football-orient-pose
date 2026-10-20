from __future__ import annotations

import importlib.util
from math import comb
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

_SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluation" / "significance_table3.py"
_spec = importlib.util.spec_from_file_location("significance_table3", _SCRIPT)
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)


def test_holm_known_values() -> None:
    # p ordenados 0,01 < 0,02 < 0,04: ×3, ×2, ×1, com monotonia
    np.testing.assert_allclose(sig.holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])
    assert sig.holm([0.6, 0.9]) == [1.0, 1.0]  # limitado a 1


def test_welch_contrast_matches_scipy_for_two_groups() -> None:
    rng = np.random.default_rng(0)
    groups = {"x": rng.normal(10, 1, 10), "y": rng.normal(9, 3, 10)}
    est, half, df, p = sig.welch_contrast(groups, {"x": 1.0, "y": -1.0})
    ref = stats.ttest_ind(groups["x"], groups["y"], equal_var=False)
    assert est == pytest.approx(groups["x"].mean() - groups["y"].mean())
    assert p == pytest.approx(ref.pvalue)
    assert df == pytest.approx(ref.df)
    assert half == pytest.approx(stats.t.ppf(0.975, df) * abs(est / ref.statistic))


def test_interaction_contrast() -> None:
    # médias A=0, B=10, C=8, D=14 → interação = 14 − 8 − 10 + 0 = −4
    base = np.array([-1.0, 0.0, 1.0])
    groups = {"A": base, "B": base + 10, "C": base + 8, "D-FULL": base + 14}
    est, half, df, p = sig.welch_contrast(groups, sig.INTERACTION)
    se = np.sqrt(4 * base.var(ddof=1) / 3)
    assert est == pytest.approx(-4.0)
    assert df == pytest.approx(8.0)  # 4 grupos iguais de n=3: gl = 4·(n−1)
    assert half == pytest.approx(stats.t.ppf(0.975, 8) * se)
    assert p < 0.05


def test_exact_mann_whitney_floor_for_full_separation() -> None:
    # 10 contra 10 sem sobreposição: p bilateral exato = 2 / C(20, 10)
    p = stats.mannwhitneyu(np.arange(10.0) + 100, np.arange(10.0), method="exact").pvalue
    assert p == pytest.approx(2 / comb(20, 10))


def test_versioned_csv_has_ten_runs_per_cell() -> None:
    cells = sig.load(sig.DEFAULT_CSV)
    assert sorted(cells) == sorted(sig.CELLS)
    assert all(len(rows) == 10 for rows in cells.values())
