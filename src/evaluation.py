"""Held-out evaluation: chronological per-patient split + held-out patients.

Design decisions (documented in docs/architecture.md):
- Chronological split: for each patient, the LAST ``TEST_FRACTION`` of usable
  days is held out. No random shuffling of overlapping windows: train and test
  never share a day, so near-duplicate lag windows cannot leak across.
- Held-out patients: every ``HELDOUT_PATIENT_MODULUS``-th patient (sorted id)
  is removed from training entirely; models are then scored on their usable
  days. This measures cold-start behavior on people the model never saw.
- All predictors (baselines and ML) are evaluated on exactly the same rows.
- Metrics: MAE primary, RMSE secondary, plus sample counts and per-patient
  MAE. Missing-data performance is reported by splitting test rows on whether
  last night's wearable values were complete.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
from src import models as M
from src import uncertainty as U
from src.features import TARGET, usable_for_training

BASELINE_KINDS = ["persistence", "personal_mean"]


# --------------------------------------------------------------- splitting --

def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-patient time-ordered split of usable rows into train/test."""
    train_parts, test_parts = [], []
    for pid, g in frame.groupby("patient_id", sort=True):
        g = g.sort_values("date")
        n = len(g)
        n_test = math.ceil(n * config.TEST_FRACTION) if n >= \
            config.MIN_TRAIN_ROWS_PER_PATIENT + 3 else 0
        if n_test:
            train_parts.append(g.iloc[:-n_test])
            test_parts.append(g.iloc[-n_test:])
        else:
            train_parts.append(g)
    train = pd.concat(train_parts, ignore_index=True) if train_parts \
        else frame.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=True) if test_parts \
        else frame.iloc[0:0]
    return train, test


def heldout_patient_mask(frame: pd.DataFrame) -> pd.Series:
    """True for rows of patients designated as held-out (unseen) patients."""
    pids = sorted(frame["patient_id"].unique())
    heldout = {p for i, p in enumerate(pids)
               if i % config.HELDOUT_PATIENT_MODULUS == 0}
    return frame["patient_id"].isin(heldout)


# --------------------------------------------------------------- metrics ----

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": round(float(np.abs(err).mean()), 4) if len(err) else None,
        "rmse": round(float(np.sqrt((err ** 2).mean())), 4) if len(err) else None,
    }


def per_patient_mae(test_frame: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    df = test_frame[["patient_id"]].copy()
    df["err"] = np.abs(np.asarray(y_pred, dtype=float) - test_frame[TARGET].to_numpy())
    return df.groupby("patient_id")["err"].agg(["count", "mean"]).reset_index() \
        .rename(columns={"count": "n_test", "mean": "mae"})


def missing_subset_metrics(test_frame: pd.DataFrame, y_pred: np.ndarray) -> dict:
    """MAE on rows whose lag-1 wearable values were complete vs incomplete."""
    sensor_lag1 = [f"{c}_lag1" for c in config.WEARABLE_COLUMNS
                   if f"{c}_lag1" in test_frame.columns]
    any_missing = test_frame[sensor_lag1].isna().any(axis=1).to_numpy()
    y_true = test_frame[TARGET].to_numpy(dtype=float)
    out = {}
    for label, mask in [("complete_last_night", ~any_missing),
                        ("incomplete_last_night", any_missing)]:
        if mask.sum() == 0:
            out[label] = {"n": 0, "mae": None}
            continue
        m = evaluate_predictions(y_true[mask], np.asarray(y_pred)[mask])
        out[label] = {"n": m["n"], "mae": m["mae"]}
    return out


# ------------------------------------------------------------- orchestration --

@dataclass
class EvaluationResult:
    model_name: str
    kind: str
    config: str
    chronological: dict = field(default_factory=dict)
    per_patient: pd.DataFrame | None = None
    missing_subsets: dict = field(default_factory=dict)
    heldout_patients: dict = field(default_factory=dict)
    train_residual_stats: dict = field(default_factory=dict)
    artifact: dict | None = None
    predictions: pd.DataFrame | None = None

    def summary_row(self) -> dict:
        row = {
            "model": self.model_name,
            "kind": self.kind,
            "config": self.config,
            **self.chronological,
            "heldout_mae": self.heldout_patients.get("mae"),
            "heldout_n": self.heldout_patients.get("n"),
            "train_resid_std": self.train_residual_stats.get("std"),
        }
        return row


def _predictions_frame(test: pd.DataFrame, y_pred: np.ndarray, name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "patient_id": test["patient_id"].to_numpy(),
        "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "model": name,
        "y_true": test[TARGET].to_numpy(dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
    })


def make_train_test(usable: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split within the non-held-out patients.

    Held-out patients never enter training; their rows are scored separately
    by :func:`evaluate_heldout_patients`.
    """
    return chronological_split(usable[~heldout_patient_mask(usable)])


def evaluate_model(name: str, artifact: dict, train: pd.DataFrame,
                   test: pd.DataFrame, keep_predictions: bool = True) -> EvaluationResult:
    """Evaluate one fitted artifact on the shared chronological test rows."""
    y_pred = M.predict(artifact, test)
    result = EvaluationResult(
        model_name=name,
        kind=artifact["kind"],
        config=artifact["config"],
        chronological=evaluate_predictions(test[TARGET].to_numpy(dtype=float), y_pred),
        per_patient=per_patient_mae(test, y_pred),
        missing_subsets=missing_subset_metrics(test, y_pred),
    )
    # Training residual stats (for the uncertainty proxy). For baselines the
    # train rows themselves; for ML models, in-sample training residuals.
    train_pred = M.predict(artifact, train)
    result.train_residual_stats = U.residual_stats(
        train[TARGET].to_numpy(dtype=float), train_pred)

    result.predictions = _predictions_frame(test, y_pred, name) \
        if keep_predictions else None
    result.artifact = artifact
    return result


def evaluate_heldout_patients(name: str, artifact: dict,
                              frame: pd.DataFrame) -> dict:
    """Score a fitted artifact on rows of patients never seen in training."""
    mask = heldout_patient_mask(frame)
    heldout_rows = usable_for_training(frame[mask]).sort_values(["patient_id", "date"])
    if heldout_rows.empty:
        return {"n": 0, "mae": None, "rmse": None, "n_patients": 0,
                "note": "no held-out patient rows available"}
    y_pred = M.predict(artifact, heldout_rows)
    res = evaluate_predictions(heldout_rows[TARGET].to_numpy(dtype=float), y_pred)
    res["n_patients"] = int(heldout_rows["patient_id"].nunique())
    return res


def fit_all_models(train_frame: pd.DataFrame, include_hgbr: bool = True) -> dict:
    """Fit every predictor on the same chronological training rows."""
    artifacts: dict[str, dict] = {}
    for kind in BASELINE_KINDS:
        artifacts[kind] = M.fit_baseline(kind, train_frame)
    for config_name in config.CONFIGURATIONS:
        artifacts[f"ridge_{config_name}"] = M.fit_sklearn("ridge", config_name,
                                                          train_frame)
    if include_hgbr:
        try:
            artifacts["hgbr_with_clinical"] = M.fit_sklearn("hgbr", "with_clinical",
                                                            train_frame)
        except Exception:  # HGBR is optional; absence is reported, never fatal
            artifacts.pop("hgbr_with_clinical", None)
    return artifacts


def run_full_evaluation(feature_frame: pd.DataFrame, include_hgbr: bool = True) -> dict:
    """End-to-end evaluation: chronological + held-out patients + missing data."""
    usable = usable_for_training(feature_frame).sort_values(["patient_id", "date"])
    if usable.empty:
        raise ValueError("no usable rows: targets require at least one prior rating")

    train, test = make_train_test(usable)
    artifacts = fit_all_models(train, include_hgbr=include_hgbr)

    results = {"split": {
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_patients": int(train["patient_id"].nunique()),
        "n_test_patients": int(test["patient_id"].nunique()) if len(test) else 0,
        "test_fraction": config.TEST_FRACTION,
        "heldout_modulus": config.HELDOUT_PATIENT_MODULUS,
    }, "models": {}, "predictions": []}

    for name, artifact in artifacts.items():
        if len(test):
            res = evaluate_model(name, artifact, train, test)
            res.heldout_patients = evaluate_heldout_patients(name, artifact, usable)
            results["models"][name] = res
            if res.predictions is not None:
                results["predictions"].append(res.predictions)
        else:
            results["models"][name] = EvaluationResult(model_name=name,
                                                       kind=artifact["kind"],
                                                       config=artifact["config"])
    results["predictions"] = pd.concat(results["predictions"], ignore_index=True) \
        if results["predictions"] else pd.DataFrame()

    # HGBR adoption rule: keep only if it beats ridge_with_clinical on test MAE.
    if include_hgbr and "hgbr_with_clinical" in results["models"] \
            and "ridge_with_clinical" in results["models"]:
        hg = results["models"]["hgbr_with_clinical"]
        rd = results["models"]["ridge_with_clinical"]
        results["hgbr_adoption"] = {
            "hgbr_test_mae": hg.chronological.get("mae"),
            "ridge_test_mae": rd.chronological.get("mae"),
            "adopted": bool(hg.chronological.get("mae") is not None
                            and rd.chronological.get("mae") is not None
                            and hg.chronological["mae"] < rd.chronological["mae"]),
        }
    else:
        results["hgbr_adoption"] = {"adopted": False,
                                    "note": "HGBR comparison not available"}
    results["artifacts"] = artifacts
    return results


def results_table(results: dict) -> pd.DataFrame:
    rows = [r.summary_row() for r in results["models"].values()]
    return pd.DataFrame(rows)
