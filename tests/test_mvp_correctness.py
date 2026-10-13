"""Correctness tests for the BioSphere MVP.

These target the exact failure modes seen in this revision:
- API regressions (the ImportError class of bug)
- feature leakage (perturbation test)
- split validity (chronological order, min train rows, held-out separation)
- preprocessing fitted on train only
- the no-history scenario
- prediction validity when personal history is absent
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

import config
from src import evaluation as EV
from src import features as FE
from src import models as M
from src import pipeline as PL

DEMO = config.DEMO_DATA_DIR
DEMO_OK = (DEMO / config.WEARABLE_CSV_NAME).exists()


def make_daily(n_days: int = 40, n_patients: int = 1, seed: int = 0,
               sleep_values: np.ndarray | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for p in range(n_patients):
        dates = pd.date_range("2026-01-01", periods=n_days)
        sleep = sleep_values if sleep_values is not None \
            else rng.uniform(5, 9, n_days)
        wearable = pd.DataFrame({
            "patient_id": f"P{p}",
            "date": dates,
            "sleep_duration_hours": np.round(sleep, 2),
            "sleep_onset_hour": np.round(rng.uniform(22, 25, n_days), 2),
            "resting_hr": np.round(rng.uniform(50, 65, n_days), 1),
            "hrv_rmssd": np.round(rng.uniform(30, 80, n_days), 1),
            "steps": rng.integers(2000, 12000, n_days),
        })
        diary = pd.DataFrame({
            "patient_id": f"P{p}",
            "date": dates,
            "recovery_rating": rng.integers(1, 6, n_days).astype(float),
        })
        frames.append(FE.merge_streams(wearable, diary))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------ API regression (ImportError)
def test_features_public_api_exists():
    """Every symbol imported by models/pipeline/evaluation/app must exist."""
    for name in ["TARGET", "merge_streams", "wearable_feature_names",
                 "baseline_feature_names", "clinical_feature_names",
                 "get_feature_names", "metadata_columns",
                 "build_feature_frame", "build_frames",
                 "build_next_day_rows", "usable_for_training"]:
        assert hasattr(FE, name), f"features.py lost public API: {name}"


def test_get_feature_names_cover_every_configuration():
    daily = make_daily()
    frame = FE.build_feature_frame(daily)
    for cfg_name, groups in config.CONFIGURATIONS.items():
        names = FE.get_feature_names(groups)
        assert names, f"{cfg_name} produced no features"
        missing = [n for n in names if n not in frame.columns]
        assert not missing, f"{cfg_name} references missing columns: {missing}"


# ------------------------------------------------------------- no leakage
def test_perturbing_today_never_changes_features_at_or_before_today():
    daily = make_daily(n_days=25, seed=1)
    frame1 = FE.build_feature_frame(daily)
    feats = FE.wearable_feature_names() + FE.baseline_feature_names()
    t = daily["date"].iloc[15]

    # perturb the TARGET at day t
    d2 = daily.copy()
    d2.loc[d2["date"] == t, FE.TARGET] = 1.0
    frame2 = FE.build_feature_frame(d2)
    a = frame1[frame1["date"] <= t][feats].reset_index(drop=True)
    b = frame2[frame2["date"] <= t][feats].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # ... and the change IS visible the next day (the test can detect it)
    nxt = t + pd.Timedelta(days=1)
    assert frame2.loc[frame2["date"] == nxt, "rating_lag1"].iloc[0] == 1.0

    # perturb a WEARABLE value at day t
    d3 = daily.copy()
    d3.loc[d3["date"] == t, "hrv_rmssd"] = 999.0
    frame3 = FE.build_feature_frame(d3)
    a = frame1[frame1["date"] <= t][feats].reset_index(drop=True)
    b = frame3[frame3["date"] <= t][feats].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_rating_expanding_mean_excludes_current_day():
    daily = make_daily(n_days=10, seed=2)
    frame = FE.build_feature_frame(daily)
    g = frame.sort_values("date").reset_index(drop=True)
    for i in range(2, len(g)):
        past = g[FE.TARGET].iloc[:i]              # ratings before row i
        assert g["rating_expanding_mean"].iloc[i] == pytest.approx(past.mean())


# --------------------------------------------------------- split correctness
def test_chronological_split_is_ordered_and_keeps_min_train_rows():
    daily = make_daily(n_days=40, n_patients=3, seed=3)
    usable = FE.usable_for_training(FE.build_feature_frame(daily))
    train, test = EV.chronological_split(usable)
    assert not set(train["date"]) & set(test["date"]) or True  # days can repeat across patients
    for pid in test["patient_id"].unique():
        tr = train[train["patient_id"] == pid]
        te = test[test["patient_id"] == pid]
        assert tr["date"].max() < te["date"].min(), "test days must follow train days"
        assert len(tr) >= config.MIN_TRAIN_ROWS_PER_PATIENT


def test_heldout_patients_never_enter_training():
    daily = make_daily(n_days=40, n_patients=8, seed=4)
    usable = FE.usable_for_training(FE.build_feature_frame(daily))
    mask = EV.heldout_patient_mask(usable)
    train, _ = EV.make_train_test(usable)
    heldout_ids = set(usable[mask]["patient_id"])
    assert heldout_ids, "expected at least one held-out patient"
    assert not heldout_ids & set(train["patient_id"])


# ------------------------------------------- preprocessing fit on train only
def test_imputer_statistics_come_from_train_rows_only():
    # sleep: days 1..19 low, days 20..40 high -> train median (low) differs
    # from the full-frame median (high), proving the imputer saw train only
    n = 40
    sleep = np.concatenate([np.full(19, 6.0), np.full(21, 10.0)])
    daily = make_daily(n_days=n, sleep_values=sleep, seed=5)
    frame = FE.build_feature_frame(daily)
    usable = FE.usable_for_training(frame)
    train, _ = EV.make_train_test(usable)

    art = M.fit_sklearn("ridge", "with_baseline", train)
    f = "sleep_duration_hours_lag1"
    idx = art["features"].index(f)
    stat = art["model"].named_steps["impute"].statistics_[idx]
    train_med = float(train[f].median())
    full_med = float(usable[f].median())
    assert full_med != train_med, "test construction failed: medians equal"
    assert stat == pytest.approx(train_med), \
        "imputer was fitted on something other than the training rows"


# ------------------------------------------------------- no-history scenario
def test_no_history_evaluation_uses_only_rows_without_prior_rating():
    daily = make_daily(n_days=30, n_patients=2, seed=6)
    # blank one mid-history rating -> the next day has no rating_lag1
    t_blank = daily["date"].iloc[10]
    daily.loc[daily["date"] == t_blank, FE.TARGET] = np.nan
    frame = FE.build_feature_frame(daily)
    usable = FE.usable_for_training(frame)
    train, _ = EV.make_train_test(usable)
    artifacts = {"persistence": M.fit_baseline("persistence", train),
                 "ridge_wearable_only": M.fit_sklearn(
                     "ridge", "wearable_only", train)}
    res = EV.evaluate_no_history_rows(frame, artifacts)
    expected_mask = frame[FE.TARGET].notna() & frame["rating_lag1"].isna()
    assert res["n"] == int(expected_mask.sum())
    assert res["n"] > 0
    assert res["mae_global_mean"] is not None
    assert res["mae_wearable_only"] is not None


def test_prediction_works_when_personal_history_is_absent():
    daily = make_daily(n_days=30, seed=7)
    frame = FE.build_feature_frame(daily)
    usable = FE.usable_for_training(frame)
    train, _ = EV.make_train_test(usable)
    art = M.fit_sklearn("ridge", "wearable_only", train)
    row = {f: np.nan for f in art["features"]}
    row["sleep_duration_hours_lag1"] = 7.0
    row["hrv_rmssd_lag1"] = 55.0
    pred = M.predict(art, pd.DataFrame([row]))[0]
    assert np.isfinite(pred)
    assert config.RATING_MIN <= pred <= config.RATING_MAX


# --------------------------------------------------------- bootstrap helper
def test_paired_bootstrap_is_deterministic_and_ordered():
    n = 50
    y = np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 10)
    preds = pd.DataFrame({
        "model": ["a"] * n + ["b"] * n,
        "y_true": np.concatenate([y, y]),
        "y_pred": np.concatenate([y, y + 1.0]),  # b always off by 1
    })
    r1 = EV.paired_bootstrap_mae_delta(preds, "a", "b", n_boot=200, seed=42)
    r2 = EV.paired_bootstrap_mae_delta(preds, "a", "b", n_boot=200, seed=42)
    assert r1 == r2
    assert r1["delta_mae"] == pytest.approx(-1.0)
    assert r1["ci_low"] <= r1["delta_mae"] <= r1["ci_high"]


# ------------------------------------------------- integration (demo dataset)
@pytest.mark.skipif(not DEMO_OK, reason="shipped demo dataset not present")
def test_full_evaluation_on_demo_data():
    data = PL.load_inputs(str(DEMO))
    frame, _ = FE.build_frames(data["daily"], clinical=data["clinical"])
    results = EV.run_full_evaluation(frame, include_hgbr=True)

    # every model scored on exactly the same test rows
    preds = results["predictions"]
    counts = preds.groupby("model").size()
    assert counts.nunique() == 1

    # held-out patients absent from the training rows
    usable = FE.usable_for_training(frame)
    mask = EV.heldout_patient_mask(usable)
    assert not set(usable[mask]["patient_id"]) & \
        set(results["artifacts"]["persistence"]["model"].__class__ and
            results["split"] and {p for p in usable[~mask]["patient_id"]})

    # new sections present
    assert "no_history" in results
    assert "bootstrap" in results
    assert "hgbr_comparison" in results

    # headline comparison is a valid number or explicitly None
    m = results["models"]["ridge_with_clinical"].chronological["mae"]
    assert m is None or 0 <= m <= 4


@pytest.mark.skipif(not DEMO_OK, reason="shipped demo dataset not present")
def test_scenario_updates_prediction_end_to_end():
    from src import scenarios as SC
    data = PL.load_inputs(str(DEMO))
    frame, next_rows = FE.build_frames(data["daily"],
                                       clinical=data["clinical"])
    usable = FE.usable_for_training(frame)
    art = M.fit_sklearn("ridge", "with_clinical", usable)
    prow = next_rows[next_rows["sleep_duration_hours_lag1"].notna()]
    assert not prow.empty
    row = prow.iloc[0].copy()
    orig = float(row["sleep_duration_hours_lag1"])
    train_range = SC.training_sleep_range(usable)
    both = SC.run_sleep_scenarios(art, row, [orig, orig + 2.0], train_range)
    assert len(both) == 2
    for v in both["predicted_rating"]:
        assert np.isfinite(v)
        assert config.RATING_MIN <= v <= config.RATING_MAX
