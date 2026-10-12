"""Tests for models: baselines, Ridge pipeline, bounds, persistence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

import config
from src import evaluation as EV
from src import models as M
from src.features import usable_for_training


@pytest.fixture(scope="module")
def split(small_feature_frame):
    return EV.make_train_test(usable_for_training(small_feature_frame))


def test_persistence_uses_last_rating(split):
    train, test = split
    art = M.fit_baseline("persistence", train)
    y = M.predict(art, test)
    expected = test["rating_lag1"].to_numpy(dtype=float)
    assert np.allclose(y, np.clip(expected, 1, 5))


def test_persistence_fallback_to_global_mean(split):
    train, test = split
    art = M.fit_baseline("persistence", train)
    # A row with no rating history must fall back to the fitted global mean.
    row = test.iloc[[0]].copy()
    row["rating_lag1"] = np.nan
    y = M.predict(art, row)
    assert y[0] == pytest.approx(np.clip(art["global_mean"], 1, 5))
    assert art["model"].fallback_mask[0]


def test_personal_mean_baseline(split):
    train, test = split
    art = M.fit_baseline("personal_mean", train)
    y = M.predict(art, test)
    assert y.min() >= 1 and y.max() <= 5
    assert len(y) == len(test)


def test_ridge_configs_fit_and_predict_in_bounds(split):
    train, test = split
    for config_name in config.CONFIGURATIONS:
        art = M.fit_sklearn("ridge", config_name, train)
        y = M.predict(art, test)
        assert len(y) == len(test)
        assert y.min() >= config.RATING_MIN - 1e-9
        assert y.max() <= config.RATING_MAX + 1e-9


def test_ridge_is_regularized_linear(split):
    train, _ = split
    art = M.fit_sklearn("ridge", "with_baseline", train)
    ridge = art["model"].named_steps["ridge"]
    assert isinstance(ridge, Ridge)
    assert ridge.alpha == config.RIDGE_ALPHA


def test_feature_count_matches_config(split):
    train, _ = split
    from src.features import get_feature_names
    for name, groups in config.CONFIGURATIONS.items():
        art = M.fit_sklearn("ridge", name, train)
        assert art["n_train_features"] == len(get_feature_names(groups))
    # strictly increasing feature counts across the ablation ladder
    assert (M.fit_sklearn("ridge", "wearable_only", train)["n_train_features"]
            < M.fit_sklearn("ridge", "with_baseline", train)["n_train_features"]
            < M.fit_sklearn("ridge", "with_clinical", train)["n_train_features"])


def test_artifact_save_load_roundtrip(split, tmp_path):
    train, test = split
    art = M.fit_sklearn("ridge", "with_clinical", train)
    path = tmp_path / "model.joblib"
    M.save_artifact(art, path, extra_meta={"note": "test"})
    loaded, meta = M.load_artifact(path)
    y1 = M.predict(art, test)
    y2 = M.predict(loaded, test)
    np.testing.assert_allclose(y1, y2)
    assert meta["kind"] == "ridge"
    assert meta["data_scope"].startswith("SYNTHETIC")


def test_unknown_baseline_kind_rejected():
    with pytest.raises(ValueError):
        M.SimpleBaseline("magic")


def test_fit_requires_usable_rows():
    empty = pd.DataFrame({"target": [np.nan], "rating_lag1": [np.nan]})
    with pytest.raises(ValueError):
        M.fit_baseline("persistence", empty)
