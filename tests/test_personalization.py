"""Tests for personal baselines (sufficiency, robust stats, no invention)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.personalization import PersonalBaseline


def _daily(n_days=20, hrv=50.0, rhr=60.0):
    rows = []
    for d in range(n_days):
        rows.append({
            "patient_id": "p1",
            "date": pd.Timestamp("2026-07-01") + pd.Timedelta(days=d),
            "sleep_duration_hours": 7.0,
            "sleep_onset_hour": 23.0,
            "resting_hr": rhr,
            "hrv_rmssd": hrv,
            "steps": 8000.0,
            config.RATING_COLUMN: 4.0,
        })
    return pd.DataFrame(rows)


def test_baseline_summary_stable_history():
    daily = _daily()
    pb = PersonalBaseline(min_obs=7).fit(daily)
    s = pb.summary("p1", "hrv_rmssd")
    assert s["status"] == "ok"
    assert s["n_obs"] == 20
    assert s["sufficient"] is True
    assert s["median"] == 50.0
    assert s["mad_sigma"] == 0.0  # constant history -> zero dispersion


def test_insufficient_history_flagged():
    daily = _daily(n_days=5)
    pb = PersonalBaseline(min_obs=7).fit(daily)
    s = pb.summary("p1", "hrv_rmssd")
    assert s["sufficient"] is False
    cmp = pb.recent_comparison(daily, "p1", "hrv_rmssd")
    assert cmp["sufficient"] is False
    assert "insufficient" in cmp["note"]


def test_missing_metric_reported_not_invented():
    daily = _daily().drop(columns=["hrv_rmssd"])
    pb = PersonalBaseline().fit(daily)
    s = pb.summary("p1", "hrv_rmssd")
    assert s is not None and s["status"] == "unavailable"
    cmp = pb.recent_comparison(daily, "p1", "hrv_rmssd")
    assert cmp["status"] == "unavailable"


def test_unknown_metric_returns_none():
    pb = PersonalBaseline().fit(_daily())
    assert pb.summary("p1", "not_a_metric") is None


def test_robust_z_detects_recent_elevation():
    daily = _daily(n_days=20)
    daily.loc[daily.index[-3:], "resting_hr"] = 75.0  # recent jump
    pb = PersonalBaseline(min_obs=7).fit(daily)
    cmp = pb.recent_comparison(daily, "p1", "resting_hr")
    assert cmp["recent_mean"] == 75.0
    assert cmp["baseline_median"] == 60.0
    # MAD is zero for a constant history; z must be explicitly unavailable.
    assert cmp["robust_z"] is None


def test_robust_z_with_dispersion():
    rng = np.random.default_rng(3)
    daily = _daily(n_days=30)
    daily["resting_hr"] = rng.normal(60.0, 4.0, size=30)
    daily.loc[daily.index[-3:], "resting_hr"] = 72.0
    pb = PersonalBaseline(min_obs=7).fit(daily)
    cmp = pb.recent_comparison(daily, "p1", "resting_hr")
    assert cmp["robust_z"] is not None and cmp["robust_z"] > 1.5


def test_completeness_counts():
    daily = _daily()
    daily.loc[daily.index[:5], "steps"] = np.nan
    pb = PersonalBaseline().fit(daily)
    comp = pb.completeness(daily, "p1")
    assert comp["steps"]["present"] == 15
    assert comp["steps"]["pct"] == 75.0
    assert comp["hrv_rmssd"]["pct"] == 100.0


def test_baseline_never_sees_data_it_was_not_given():
    """Fit on the first 10 days; a recent_comparison on later days must not
    silently extend the baseline statistics (n stays 10)."""
    daily = _daily(n_days=20)
    pb = PersonalBaseline().fit(daily.iloc[:10])
    s = pb.summary("p1", "resting_hr")
    assert s["n_obs"] == 10
    cmp = pb.recent_comparison(daily, "p1", "resting_hr")
    assert cmp["n_baseline"] == 10
