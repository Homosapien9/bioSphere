"""Forecasting models: two transparent baselines + regularized regression.

Model zoo (all deterministic, all evaluated on identical held-out rows):
  persistence    : predict tomorrow's rating = today's (most recent) rating.
  personal_mean  : predict with the patient's expanding mean of past ratings.
  ridge_<config> : scikit-learn Pipeline (median impute -> standardize ->
                   Ridge) over the config's feature set. Imputation and
                   scaling are fitted on training rows only, preventing
                   leakage.
  hgbr_with_clinical (optional comparison): HistGradientBoostingRegressor,
                   adopted ONLY if it actually improves held-out MAE.

Every model output is clipped to the 1-5 rating scale, which is the scale the
label was defined on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.features import get_feature_names, usable_for_training


def _clip(v: np.ndarray) -> np.ndarray:
    return np.clip(v, config.RATING_MIN, config.RATING_MAX)


def make_ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=config.RIDGE_ALPHA)),
    ])


def make_hgbr_pipeline() -> Pipeline:
    # Trees handle NaN natively; imputation would only blur missingness patterns.
    return Pipeline([
        ("hgbr", HistGradientBoostingRegressor(random_state=config.RANDOM_SEED)),
    ])


class SimpleBaseline:
    """Persistence / personal-expanding-mean predictors with a fitted fallback.

    Fallback: when a patient has no prior rating at all, both baselines fall
    back to the global mean of training targets (fitted on train rows only).
    The affected rows are countable via ``fallback_mask``.
    """

    def __init__(self, kind: str):
        if kind not in ("persistence", "personal_mean"):
            raise ValueError(f"unknown baseline kind '{kind}'")
        self.kind = kind
        self.global_mean: float | None = None
        self.fallback_mask: np.ndarray | None = None

    def fit(self, train_frame: pd.DataFrame) -> "SimpleBaseline":
        targets = train_frame["target"].dropna()
        if targets.empty:
            raise ValueError("training frame has no usable targets")
        self.global_mean = float(targets.mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.kind == "persistence":
            col = frame["rating_lag1"].to_numpy(dtype=float)
        else:
            col = frame["rating_expanding_mean"].to_numpy(dtype=float)
        self.fallback_mask = np.isnan(col)
        col = np.where(self.fallback_mask, self.global_mean, col)
        return _clip(col)


def fit_sklearn(kind: str, config_name: str, train_frame: pd.DataFrame) -> dict:
    """Fit the sklearn-backed model (``ridge`` or ``hgbr``) for one ablation
    configuration. Returns an artifact dict consumed by :func:`predict`.
    """
    if kind not in ("ridge", "hgbr"):
        raise ValueError(f"unknown sklearn model kind '{kind}'")
    features = get_feature_names(config.CONFIGURATIONS[config_name])
    missing = [f for f in features if f not in train_frame.columns]
    if missing:
        raise ValueError(f"feature frame lacks columns required by "
                         f"'{config_name}': {missing[:5]}")
    usable = usable_for_training(train_frame)
    if usable.empty:
        raise ValueError(f"no usable training rows for '{config_name}'")
    pipe = make_ridge_pipeline() if kind == "ridge" else make_hgbr_pipeline()
    pipe.fit(usable[features], usable["target"])
    return {"kind": kind, "config": config_name, "features": features,
            "model": pipe, "n_train_rows": int(len(usable)),
            "n_train_features": len(features)}


def fit_baseline(kind: str, train_frame: pd.DataFrame) -> dict:
    model = SimpleBaseline(kind).fit(train_frame)
    return {"kind": kind, "config": "baseline", "features": [],
            "model": model, "n_train_rows": int(len(train_frame)),
            "n_train_features": 0,
            "global_mean": model.global_mean}


def predict(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict with any fitted artifact; outputs are clipped to [1, 5]."""
    model = artifact["model"]
    if artifact["kind"] in ("persistence", "personal_mean"):
        return model.predict(frame)
    return _clip(np.asarray(model.predict(frame[artifact["features"]]), dtype=float))


# ------------------------------------------------------------- persistence --

def save_artifact(artifact: dict, path, extra_meta: dict | None = None) -> None:
    """Persist a fitted artifact (joblib) with provenance metadata."""
    import joblib

    meta = {
        "project": config.PROJECT_NAME,
        "version": config.PROJECT_VERSION,
        "kind": artifact["kind"],
        "config": artifact["config"],
        "features": artifact["features"],
        "n_train_rows": artifact["n_train_rows"],
        "random_seed": config.RANDOM_SEED,
        "data_scope": "SYNTHETIC — generated by src/data_generation.py",
    }
    if extra_meta:
        meta.update(extra_meta)
    joblib.dump({"artifact": artifact, "metadata": meta}, path)


def load_artifact(path) -> tuple[dict, dict]:
    import joblib

    payload = joblib.load(path)
    return payload["artifact"], payload["metadata"]
