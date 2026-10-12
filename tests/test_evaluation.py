"""Tests for evaluation: split integrity, metric math, missing-data subsets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import evaluation as EV
from src.features import usable_for_training


def test_mae_matches_manual_computation():
    y = np.array([1.0, 2.0, 5.0])
    p = np.array([1.5, 2.0, 3.0])
    res = EV.evaluate_predictions(y, p)
    assert res["n"] == 3
    # evaluate_predictions reports at 4-dp precision.
    assert res["mae"] == pytest.approx((0.5 + 0.0 + 2.0) / 3, abs=1e-4)
    assert res["rmse"] == pytest.approx(np.sqrt((0.25 + 0.0 + 4.0) / 3), abs=1e-4)


def test_chronological_split_respects_time(small_feature_frame):
    usable = usable_for_training(small_feature_frame)
    trs, tes = [], []
    for _, g in usable.groupby("patient_id"):
        tr, te = EV.chronological_split(g)
        trs.append(tr)
        if len(te):
            tes.append(te)
    train, test = pd.concat(trs), pd.concat(tes)
    assert len(test) > 0
    max_train = train.groupby("patient_id")["date"].max()
    min_test = test.groupby("patient_id")["date"].min()
    for pid in min_test.index:
        if pid in max_train.index:
            assert max_train[pid] < min_test[pid], \
                f"train/test overlap for {pid}"
    # test share approximates TEST_FRACTION
    frac = len(test) / (len(test) + len(train))
    assert 0.1 < frac < 0.45


def test_small_patients_stay_in_train():
    frame = pd.DataFrame({
        "patient_id": ["a"] * 5,
        "date": pd.date_range("2026-07-01", periods=5),
        "target": [3.0] * 5,
        "rating_lag1": [3.0] * 5,
    })
    train, test = EV.chronological_split(frame)
    assert len(test) == 0  # below MIN_TRAIN_ROWS_PER_PATIENT + 3
    assert len(train) == 5


def test_heldout_patients_are_disjoint(small_feature_frame):
    usable = usable_for_training(small_feature_frame)
    mask = EV.heldout_patient_mask(usable)
    held = set(usable[mask]["patient_id"])
    rest = set(usable[~mask]["patient_id"])
    assert held and rest
    assert not (held & rest)
    # modulus rule respected
    pids = sorted(usable["patient_id"].unique())
    expected = {p for i, p in enumerate(pids)
                if i % config.HELDOUT_PATIENT_MODULUS == 0}
    assert held == expected


def test_make_train_test_excludes_heldout_patients(small_feature_frame):
    """The production split: held-out patients appear in NEITHER train nor
    test, and each test patient's earliest test day is after all their train
    days."""
    usable = usable_for_training(small_feature_frame)
    train, test = EV.make_train_test(usable)
    held = set(usable[EV.heldout_patient_mask(usable)]["patient_id"])
    assert held
    assert not (set(train["patient_id"]) & held)
    assert not (set(test["patient_id"]) & held)
    assert len(test) > 0
    for pid, g in test.groupby("patient_id"):
        tr_max = train.loc[train["patient_id"] == pid, "date"].max()
        assert g["date"].min() > tr_max, f"train/test overlap for {pid}"


def test_missing_subset_metrics_partition_test_rows(small_feature_frame):
    usable = usable_for_training(small_feature_frame)
    train, test = EV.make_train_test(usable)
    artifacts = EV.fit_all_models(train, include_hgbr=False)
    res = EV.evaluate_model("ridge_with_baseline", artifacts["ridge_with_baseline"],
                            train, test, keep_predictions=False)
    subsets = res.missing_subsets
    total = subsets["complete_last_night"]["n"] + \
        subsets["incomplete_last_night"]["n"]
    assert total == len(test)


def test_full_evaluation_structure(small_feature_frame):
    results = EV.run_full_evaluation(small_feature_frame, include_hgbr=False)
    assert results["split"]["n_test_rows"] > 0
    expected_models = {"persistence", "personal_mean",
                       "ridge_wearable_only", "ridge_with_baseline",
                       "ridge_with_clinical"}
    assert expected_models <= set(results["models"])
    table = EV.results_table(results)
    assert {"model", "mae", "rmse", "n"} <= set(table.columns)
    assert (table["n"] == results["split"]["n_test_rows"]).all()
    # All models evaluated on identical rows -> same n.
    assert results["split"]["n_test_rows"] == \
        results["models"]["persistence"].chronological["n"]


def test_hgbr_adoption_rule_prefers_simpler_model_on_tie(small_feature_frame):
    results = EV.run_full_evaluation(small_feature_frame, include_hgbr=True)
    adoption = results["hgbr_adoption"]
    if adoption.get("adopted"):
        assert adoption["hgbr_test_mae"] < adoption["ridge_test_mae"]


def test_predictions_frame_columns(small_feature_frame):
    results = EV.run_full_evaluation(small_feature_frame, include_hgbr=False)
    preds = results["predictions"]
    assert {"patient_id", "date", "model", "y_true", "y_pred"} <= set(preds.columns)
    assert preds["y_pred"].between(1, 5).all()
