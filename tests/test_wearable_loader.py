"""Tests for the wearable/diary CSV loaders."""
from __future__ import annotations

import pandas as pd
import pytest

import config
from src import wearable_loader as WL


def test_demo_roundtrip(loaded_demo):
    w = loaded_demo["wearable"]
    assert len(w) == 6 * 45
    assert str(w["date"].dtype).startswith("datetime64")
    assert loaded_demo["w_report"].rows == len(w)
    assert loaded_demo["w_report"].patients == 6


def test_non_numeric_values_masked_and_reported(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text(
        "patient_id,date,sleep_duration_hours,sleep_onset_hour,resting_hr,"
        "hrv_rmssd,steps\n"
        "p1,2026-07-01,7.5,23.2,60,50,8000\n"
        "p1,2026-07-02,not-a-number,23.2,60,50,8000\n"
        "p1,2026-07-03,7.0,23.0,60,50,-9999\n"
    )
    df, report = WL.load_wearable_csv(p)
    assert pd.isna(df.loc[1, "sleep_duration_hours"])
    assert report.coerced_to_nan.get("sleep_duration_hours") == 1
    # -9999 steps is out of plausible range: masked and reported, not silently kept
    assert pd.isna(df.loc[2, "steps"])
    assert report.masked_implausible.get("steps") == 1


def test_absent_sensor_column_becomes_missing_not_invented(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("patient_id,date,steps\np1,2026-07-01,8000\n")
    df, report = WL.load_wearable_csv(p)
    assert df["hrv_rmssd"].isna().all()
    assert df["steps"].iloc[0] == 8000
    assert "hrv_rmssd (column absent)" in report.coerced_to_nan


def test_missing_patient_id_column_raises(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("date,steps\n2026-07-01,8000\n")
    with pytest.raises(ValueError, match="patient_id"):
        WL.load_wearable_csv(p)


def test_bad_date_dropped_and_reported(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text(
        "patient_id,date,steps\n"
        "p1,2026-07-01,8000\n"
        "p1,not-a-date,8000\n"
    )
    df, report = WL.load_wearable_csv(p)
    assert len(df) == 1
    assert report.coerced_to_nan.get("date") == 1


def test_diary_out_of_range_ratings_reported(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(f"patient_id,date,{config.RATING_COLUMN}\n"
                 "p1,2026-07-01,3\n"
                 "p1,2026-07-02,9\n"
                 "p1,2026-07-03,2.5\n")
    df, report = WL.load_diary_csv(p)
    assert pd.isna(df[config.RATING_COLUMN].iloc[1])
    assert pd.isna(df[config.RATING_COLUMN].iloc[2])
    assert report.masked_implausible


def test_diary_sorted_by_patient_and_date(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(f"patient_id,date,{config.RATING_COLUMN}\n"
                 "p2,2026-07-02,4\np1,2026-07-01,3\np1,2026-07-03,5\n")
    df, _ = WL.load_diary_csv(p)
    assert list(df["patient_id"]) == ["p1", "p1", "p2"]
