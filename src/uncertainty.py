"""Uncertainty proxy from held-back training residuals.

Biosphere does NOT produce calibrated prediction intervals. What it provides
is an honest, clearly labeled proxy: the distribution of residuals the model
achieved on training data (or, preferably, on held-out folds), summarized as a
standard deviation (plus empirical 2.5/97.5 percentiles in metrics.json).
The ±1.96σ band is labeled in the UI as "error visualization", following
BioTwin's wording discipline — it is not a statistical guarantee.
"""
from __future__ import annotations

import numpy as np

import config


def residual_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true
    resid = resid[np.isfinite(resid)]
    if len(resid) == 0:
        return {"n": 0, "std": None, "q025": None, "q975": None,
                "mae": None, "bias": None}
    return {
        "n": int(len(resid)),
        "std": round(float(resid.std(ddof=1)), 4) if len(resid) > 1 else None,
        "q025": round(float(np.percentile(resid, 2.5)), 4),
        "q975": round(float(np.percentile(resid, 97.5)), 4),
        "mae": round(float(np.abs(resid).mean()), 4),
        "bias": round(float(resid.mean()), 4),
    }


def gaussian_interval(yhat: float, stats: dict) -> tuple[float, float] | None:
    """±1.96σ band clipped to the rating scale; None when σ is unavailable."""
    if stats.get("std") is None:
        return None
    half = 1.96 * stats["std"]
    lo = max(config.RATING_MIN, yhat - half)
    hi = min(config.RATING_MAX, yhat + half)
    return round(lo, 2), round(hi, 2)

