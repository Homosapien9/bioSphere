"""Bounded what-if scenario explorer: vary tonight's sleep, hold the rest fixed.

The ONLY scenario feature in v1. It works on the actual fitted model: the
selected feature row is copied, the sleep-related features are recomputed
consistently (lag, deviation from the personal historical mean, 7-day rolling
mean, expanding historical mean), and the model is re-run. Nothing is
hard-coded animation or an arbitrary score change.

Outputs are labeled HYPOTHETICAL MODEL SCENARIOS. They are not causal
estimates: the model captures associations in synthetic data, and holding all
other features fixed while changing one is a modeling assumption. Scenarios
outside the model's training range are flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import models as M
from src import uncertainty as U

ROLL = config.ROLLING_WINDOW


def training_sleep_range(train_frame: pd.DataFrame) -> tuple[float, float]:
    """Plausible-input bounds from the training distribution of sleep lag-1."""
    col = "sleep_duration_hours_lag1"
    if col not in train_frame.columns or train_frame[col].notna().sum() == 0:
        return (float(config.WEARABLE_RANGES["sleep_duration_hours"][0]),
                float(config.WEARABLE_RANGES["sleep_duration_hours"][1]))
    low, high = config.SCENARIO_TRAIN_RANGE_PERCENTILES
    lo = float(np.percentile(train_frame[col].dropna(), low))
    hi = float(np.percentile(train_frame[col].dropna(), high))
    return lo, hi


def apply_sleep_scenario(base_row: pd.Series, new_sleep_hours: float) -> pd.Series:
    """Return a copy of ``base_row`` with tonight's sleep set to the scenario.

    Dependent features are recomputed so the model sees a coherent input:
      sleep_duration_hours_lag1      -> new value
      sleep_duration_hours_dev_lag1  -> new - (personal expanding mean after the change)
      sleep_duration_hours_hist_mean -> old mean + (new - old) / n_hist
      sleep_duration_hours_roll7_mean-> old roll7 + (new - old) / 7
    """
    row = base_row.copy()
    col_lag1 = "sleep_duration_hours_lag1"
    old = row.get(col_lag1)
    if old is None or (isinstance(old, float) and np.isnan(old)):
        raise ValueError(
            "the base row has no recorded last-night sleep to modify — "
            "scenario comparison requires an observed starting value")
    old = float(old)
    lo, hi = config.SCENARIO_SLEEP_RANGE
    new = float(np.clip(new_sleep_hours, lo, hi))
    if not (lo <= float(new_sleep_hours) <= hi):
        row["scenario_clipped"] = True  # recorded; caller should surface it

    row[col_lag1] = new
    row[f"{col_lag1}_missing"] = 0.0

    n_hist = row.get("sleep_duration_hours_hist_n")
    n_hist = float(n_hist) if pd.notna(n_hist) else 1.0
    old_hist = row.get("sleep_duration_hours_hist_mean")
    if pd.notna(old_hist):
        new_hist = float(old_hist) + (new - old) / max(n_hist, 1.0)
        row["sleep_duration_hours_hist_mean"] = new_hist
        row["sleep_duration_hours_dev_lag1"] = new - new_hist
    else:
        # No personal history: deviation undefined; keep it missing and let the
        # model's imputer treat it as "unremarkable" (its training median).
        row["sleep_duration_hours_dev_lag1"] = np.nan
        row["sleep_duration_hours_hist_mean"] = np.nan

    old_roll = row.get("sleep_duration_hours_roll7_mean")
    if pd.notna(old_roll):
        row["sleep_duration_hours_roll7_mean"] = float(old_roll) + (new - old) / ROLL
    return row


def run_sleep_scenarios(artifact: dict, base_row: pd.Series,
                        sleep_values: list[float],
                        train_range: tuple[float, float],
                        resid_stats: dict | None = None) -> pd.DataFrame:
    """Predict the rating under each hypothetical sleep value.

    Returns a frame with one row per scenario: the sleep value, whether it is
    inside the training range, the model's prediction, and the uncertainty
    proxy interval when residual statistics are supplied. All scenario rows are
    predicted in a single matrix call (identical results: every pipeline step
    is row-wise).
    """
    features = artifact["features"]
    scenario_rows = [apply_sleep_scenario(base_row, s) for s in sleep_values]
    X = pd.DataFrame([r[features] for r in scenario_rows], columns=features)
    preds = M.predict(artifact, X)

    recs = []
    for sleep, pred in zip(sleep_values, preds):
        rec = {
            "sleep_hours": round(float(sleep), 2),
            "predicted_rating": round(float(pred), 3),
            "in_training_range": bool(train_range[0] <= sleep <= train_range[1]),
            "interval_low": None,
            "interval_high": None,
        }
        if resid_stats:
            interval = U.gaussian_interval(float(pred), resid_stats)
            if interval:
                rec["interval_low"], rec["interval_high"] = interval
        recs.append(rec)
    return pd.DataFrame(recs)


def scenario_note() -> str:
    return (
        "What-if outputs are HYPOTHETICAL MODEL SCENARIOS: the fitted model is "
        "re-run with tonight's sleep replaced and every other input held fixed. "
        "This is an association in synthetic data, not a causal effect and not "
        "medical advice. Real behavior would also change HRV, resting heart "
        "rate, and activity — the model does not simulate those knock-on "
        "effects, which this fixed-input design deliberately avoids pretending "
        "to know."
    )
