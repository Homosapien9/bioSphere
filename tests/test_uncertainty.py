"""Tests for the residual-based uncertainty proxy."""
from __future__ import annotations

import numpy as np
import pytest

from src import uncertainty as U


def test_residual_stats_basic():
    y = np.array([3.0, 3.0, 3.0, 3.0])
    p = np.array([2.5, 2.5, 3.5, 3.5])
    s = U.residual_stats(y, p)
    assert s["n"] == 4
    assert s["mae"] == pytest.approx(0.5)
    assert s["bias"] == pytest.approx(0.0)
    assert s["std"] == pytest.approx(np.std([0.5, 0.5, -0.5, -0.5], ddof=1), abs=1e-4)


def test_gaussian_interval_clipped_to_scale():
    stats = {"std": 3.0}  # huge sigma
    lo, hi = U.gaussian_interval(4.5, stats)
    assert lo == 1.0 and hi == 5.0
    lo, hi = U.gaussian_interval(3.0, {"std": 0.2})
    assert lo == pytest.approx(3.0 - 1.96 * 0.2, abs=1e-2)
    assert lo >= 1.0 and hi <= 5.0


def test_missing_std_returns_none():
    assert U.gaussian_interval(3.0, {"std": None}) is None
    assert U.gaussian_interval(3.0, {}) is None
