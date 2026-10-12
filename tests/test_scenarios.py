"""Tests for the bounded sleep what-if scenario engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src import evaluation as EV
from src import models as M
from src import scenarios as S
from src.features import build_next_day_rows, usable_for_training


@pytest.fixture(scope="module")
def fitted(small_feature_frame):
    train, test = EV.make_train_test(usable_for_training(small_feature_frame))
    artifact = M.fit_sklearn("ridge", "with_clinical", train)
    return artifact, train, test


@pytest.fixture(scope="module")
def base_row(small_daily, small_clinical, fitted):
    artifact, train, _ = fitted
    rows = build_next_day_rows(small_daily, clinical=small_clinical)
    row = rows.iloc[0]
    assert pd.notna(row["sleep_duration_hours_lag1"]), \
        "first patient must have a recorded last-night sleep for scenarios"
    return row


def test_scenario_changes_model_output(fitted, base_row):
    artifact, train, _ = fitted
    train_range = S.training_sleep_range(train)
    out = S.run_sleep_scenarios(artifact, base_row,
                                [4.0, 7.0, 10.0], train_range)
    preds = out["predicted_rating"].to_numpy()
    assert len(out) == 3
    # The model is not guaranteed monotone in sleep, but outputs must differ.
    assert len(set(np.round(preds, 4))) >= 2


def test_scenario_recomputes_derived_features(base_row):
    old = float(base_row["sleep_duration_hours_lag1"])
    new_row = S.apply_sleep_scenario(base_row, old + 2.0)
    assert float(new_row["sleep_duration_hours_lag1"]) == pytest.approx(old + 2.0)
    # roll7 mean must shift by (new-old)/7
    delta = (float(new_row["sleep_duration_hours_roll7_mean"])
             - float(base_row["sleep_duration_hours_roll7_mean"]))
    assert delta == pytest.approx(2.0 / config.ROLLING_WINDOW, abs=1e-9)
    # deviation must equal new value minus updated personal mean
    dev = float(new_row["sleep_duration_hours_dev_lag1"])
    hist = float(new_row["sleep_duration_hours_hist_mean"])
    assert dev == pytest.approx(old + 2.0 - hist, abs=1e-9)


def test_scenario_outside_training_range_is_flagged(fitted, base_row):
    artifact, train, _ = fitted
    train_range = S.training_sleep_range(train)
    out = S.run_sleep_scenarios(artifact, base_row, [2.0, 12.0], train_range)
    flagged = out[~out["in_training_range"]]
    assert len(flagged) >= 1
    # a value beyond the hard scenario bounds is clipped and still predicted
    assert out["predicted_rating"].between(1, 5).all()


def test_scenario_inside_training_range_not_flagged(fitted, base_row):
    artifact, train, _ = fitted
    train_range = S.training_sleep_range(train)
    mid = float(np.mean(train_range))
    out = S.run_sleep_scenarios(artifact, base_row, [mid], train_range)
    assert out.iloc[0]["in_training_range"]


def test_scenario_is_deterministic(fitted, base_row):
    artifact, train, _ = fitted
    train_range = S.training_sleep_range(train)
    a = S.run_sleep_scenarios(artifact, base_row, [5.0, 6.5], train_range)
    b = S.run_sleep_scenarios(artifact, base_row, [5.0, 6.5], train_range)
    pd.testing.assert_frame_equal(a, b)


def test_scenario_matrix_predict_matches_manual_recompute(fitted, base_row):
    """Regression guard: the single-matrix prediction path used by
    run_sleep_scenarios must equal per-row recomputation."""
    artifact, train, _ = fitted
    train_range = S.training_sleep_range(train)
    values = [4.0, 6.5, 9.0]
    out = S.run_sleep_scenarios(artifact, base_row, values, train_range)
    for sleep, got in zip(values, out["predicted_rating"]):
        row = S.apply_sleep_scenario(base_row, sleep)
        X = pd.DataFrame([row[artifact["features"]].to_dict()],
                         columns=artifact["features"])
        # scenario frames store 3-dp display values
        assert got == pytest.approx(round(float(M.predict(artifact, X)[0]), 3),
                                    abs=1e-9)


def test_scenario_with_uncertainty_interval(fitted, base_row):
    artifact, train, _ = fitted
    train_range = S.training_sleep_range(train)
    stats = {"std": 0.8, "q025": -1.5, "q975": 1.5}
    out = S.run_sleep_scenarios(artifact, base_row, [6.0], train_range,
                                resid_stats=stats)
    r = out.iloc[0]
    assert r["interval_low"] is not None and r["interval_high"] is not None
    assert r["interval_low"] <= r["predicted_rating"] <= r["interval_high"]


def test_scenario_requires_observed_starting_value(fitted):
    artifact, train, _ = fitted
    empty_row = pd.Series({"sleep_duration_hours_lag1": np.nan})
    with pytest.raises(ValueError, match="starting value"):
        S.apply_sleep_scenario(empty_row, 7.0)


def test_scenario_note_contains_disclaimers():
    note = S.scenario_note()
    assert "HYPOTHETICAL" in note
    assert "not a causal effect" in note
    assert "not medical advice" in note
