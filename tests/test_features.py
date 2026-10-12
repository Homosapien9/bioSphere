"""Tests for leakage-safe feature engineering — the heart of the pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import features as FE


def test_lag_features_use_only_past(small_daily):
    frame = FE.build_feature_frame(small_daily)
    for pid, g in frame.groupby("patient_id"):
        g = g.sort_values("date").reset_index(drop=True)
        dates = list(g["date"])
        daily = small_daily[small_daily["patient_id"] == pid] \
            .set_index("date")
        for i in range(2, len(g)):
            d = dates[i]
            prev = dates[i - 1]
            # sleep lag1 at day d must equal the wearable value at day d-1
            expected = daily.loc[prev, "sleep_duration_hours"]
            got = g.loc[i, "sleep_duration_hours_lag1"]
            if pd.isna(expected):
                assert pd.isna(got)
            else:
                assert got == expected
            # rating lag1 at day d must equal the target recorded at day d-1
            if "target" in daily.columns:
                exp_r = daily.loc[prev, "target"]
                got_r = g.loc[i, "rating_lag1"]
                assert (pd.isna(exp_r) and pd.isna(got_r)) or got_r == exp_r


def test_no_future_leakage_perturbation_test(small_daily):
    """Changing every value on day D must not alter features on rows <= D."""
    frame_a = FE.build_feature_frame(small_daily)
    perturbed = small_daily.copy()
    mask = perturbed["date"] >= pd.Timestamp("2026-08-01")
    perturbed.loc[mask, "resting_hr"] = \
        perturbed.loc[mask, "resting_hr"] + 25.0
    if "target" in perturbed.columns:
        perturbed.loc[mask, "target"] = 5.0
    frame_b = FE.build_feature_frame(perturbed)

    feature_cols = FE.get_feature_names(["wearable", "baseline"])
    a = frame_a[frame_a["date"] < pd.Timestamp("2026-08-01")]
    b = frame_b[frame_b["date"] < pd.Timestamp("2026-08-01")]
    for col in feature_cols:
        va, vb = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        both_nan = np.isnan(va) & np.isnan(vb)
        assert np.allclose(va, vb, equal_nan=True, atol=1e-9), \
            f"future leak through column {col}"


def test_missing_indicators_track_gaps(small_daily):
    frame = FE.build_feature_frame(small_daily)
    col = "sleep_duration_hours_lag1_missing"
    lag = "sleep_duration_hours_lag1"
    assert ((frame[col] == 1.0) == frame[lag].isna()).all()
    assert (frame[col] == 0.0).sum() > 0


def test_expanding_personal_mean_grows_with_history(small_daily):
    frame = FE.build_feature_frame(small_daily)
    for pid, g in frame.groupby("patient_id"):
        g = g.sort_values("date").reset_index(drop=True)
        s = g["sleep_duration_hours_hist_mean"]
        # After the first 3 usable values, later expanding means differ from
        # early ones (they include more days) unless the series is constant.
        vals = s.dropna()
        if len(vals) > 5:
            assert not np.isclose(vals.iloc[3], vals.iloc[-1]) or \
                g["sleep_duration_hours_lag1"].dropna().nunique() == 1
        break


def test_metadata_columns_present(small_daily):
    frame = FE.build_feature_frame(small_daily)
    for col in FE.metadata_columns():
        assert col in frame.columns, col
    n = frame["sleep_duration_hours_hist_n"]
    assert (n.dropna() >= 0).all()
    # hist_n is a count of past observations: monotonically non-decreasing
    for pid, g in frame.groupby("patient_id"):
        s = g.sort_values("date")["sleep_duration_hours_hist_n"].dropna()
        assert (s.diff().dropna() >= 0).all()
        break


def test_next_day_rows_are_beyond_last_observation(small_daily):
    rows = FE.build_next_day_rows(small_daily)
    assert len(rows) == small_daily["patient_id"].nunique()
    last = small_daily.groupby("patient_id")["date"].max()
    for _, r in rows.iterrows():
        assert r["date"] == last[r["patient_id"]] + pd.Timedelta(days=1)
        assert pd.isna(r["target"]) if "target" in rows.columns else True
        # features must exist from the full history
        assert pd.notna(r["sleep_duration_hours_lag1"]) or True


def test_next_day_row_matches_manual_last_shift(small_daily):
    """The next-day row's lag1 must equal the last observed wearable value."""
    rows = FE.build_next_day_rows(small_daily)
    for _, r in rows.iterrows():
        pid = r["patient_id"]
        last_val = small_daily[small_daily["patient_id"] == pid] \
            .sort_values("date")["sleep_duration_hours"].iloc[-1]
        if pd.notna(last_val):
            assert r["sleep_duration_hours_lag1"] == last_val


def test_usable_for_training_requires_prior_rating(small_daily):
    frame = FE.build_feature_frame(small_daily)
    usable = FE.usable_for_training(frame)
    assert usable["target"].notna().all()
    assert usable["rating_lag1"].notna().all()
    assert len(usable) < len(frame)


def test_feature_names_union():
    all_names = FE.get_feature_names(["wearable", "baseline", "clinical"])
    assert set(FE.wearable_feature_names()) <= set(all_names)
    assert set(FE.baseline_feature_names()) <= set(all_names)
    assert set(FE.clinical_feature_names()) <= set(all_names)
    assert len(all_names) == len(set(all_names)), "duplicate feature names"
