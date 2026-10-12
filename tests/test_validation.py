"""Tests for validation gates."""
from __future__ import annotations

import pandas as pd
import pytest

import config
from src import validation as V


def _wearable_frame(rows=10, patients=2, value=7.0):
    out = []
    for p in range(patients):
        for d in range(rows):
            out.append({
                "patient_id": f"p{p}",
                "date": pd.Timestamp("2026-07-01") + pd.Timedelta(days=d),
                **{c: value for c in config.WEARABLE_COLUMNS},
            })
    return pd.DataFrame(out)


def test_clean_frame_passes():
    rep = V.validate_wearable(_wearable_frame())
    assert rep.passed, rep.errors


def test_missing_column_is_fatal():
    df = _wearable_frame().drop(columns=["hrv_rmssd"])
    rep = V.validate_wearable(df)
    assert not rep.passed
    assert any("hrv_rmssd" in e for e in rep.errors)


def test_duplicate_rows_are_fatal():
    df = pd.concat([_wearable_frame(rows=3), _wearable_frame(rows=1)],
                   ignore_index=True)
    rep = V.validate_wearable(df)
    assert not rep.passed
    assert any("duplicate" in e for e in rep.errors)


def test_out_of_range_value_is_warning():
    df = _wearable_frame()
    df.loc[0, "resting_hr"] = 250.0
    rep = V.validate_wearable(df)
    assert rep.passed  # plausibility issues warn; loaders mask them loudly
    assert any("resting_hr" in w for w in rep.warnings)


def test_high_missingness_warns():
    df = _wearable_frame()
    df.loc[:6, "steps"] = float("nan")
    rep = V.validate_wearable(df)
    assert any("steps" in w and "missing" in w for w in rep.warnings)


def test_diary_bad_rating_is_fatal():
    df = pd.DataFrame({
        "patient_id": ["p1"] * 3,
        "date": pd.date_range("2026-07-01", periods=3),
        config.RATING_COLUMN: [3.0, 9.0, 2.0],
    })
    rep = V.validate_diary(df)
    assert not rep.passed


def test_alignment_warns_on_missing_streams():
    w = _wearable_frame(patients=3)
    d = w[w["patient_id"] != "p2"][["patient_id", "date"]].copy()
    d[config.RATING_COLUMN] = 3.0
    clinical = pd.DataFrame({"patient_id": ["p1", "p2", "p3"]})
    rep = V.validate_alignment(w, d, clinical)
    assert any("no diary" in x for x in rep.warnings)


def test_require_passed_raises():
    rep = V.ValidationReport(errors=["boom"])
    with pytest.raises(V.DataValidationError):
        V.require_passed(rep, "test")
    V.require_passed(V.ValidationReport(), "test")  # no raise
