"""Leakage-safe chronological feature engineering.

Row contract: for the row dated ``t`` (the day whose recovery rating is the
target), every feature is computed strictly from records on days < ``t``.
Concretely, wearable/diary values are shifted by one day BEFORE any rolling
or expanding statistic is applied (shift-then-expand), so a feature at row
``t`` only ever sees data through day ``t-1``. This is enforced by a
perturbation test in tests/test_mvp_correctness.py.

Feature groups (cumulative):
  wearable : lag1/lag2, 7-day rolling mean, personal expanding mean,
             deviation of last night from the personal mean, missing indicators.
  baseline : recovery-rating lags, personal expanding rating mean, and the
             deviation of the last rating from the patient's prior mean.
  clinical : static FHIR-derived context.

Public API (imported by models.py, evaluation.py, pipeline.py, app.py —
do not remove or rename these):
  TARGET, merge_streams, wearable_feature_names, baseline_feature_names,
  clinical_feature_names, get_feature_names, metadata_columns,
  build_feature_frame, build_frames, build_next_day_rows, usable_for_training
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import config
from src.fhir_loader import CLINICAL_FEATURES  # canonical list, re-exported

TARGET = "target"


def merge_streams(wearable: pd.DataFrame, diary: pd.DataFrame) -> pd.DataFrame:
    """Outer-join wearable and diary on (patient_id, date), sorted chronologically."""
    daily = wearable.merge(diary, on=["patient_id", "date"], how="outer")
    daily = daily.sort_values(["patient_id", "date"]).reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.rename(columns={config.RATING_COLUMN: TARGET})
    return daily


def wearable_feature_names() -> list[str]:
    names = []
    for col in config.WEARABLE_COLUMNS:
        names += [f"{col}_lag1", f"{col}_lag2", f"{col}_roll7_mean",
                  f"{col}_hist_mean", f"{col}_dev_lag1", f"{col}_lag1_missing"]
    return names


def baseline_feature_names() -> list[str]:
    return ["rating_lag1", "rating_lag2", "rating_expanding_mean",
            "rating_dev_lag1", "rating_lag1_missing"]


def clinical_feature_names() -> list[str]:
    return list(CLINICAL_FEATURES)


def get_feature_names(groups: list[str]) -> list[str]:
    """Feature names for a cumulative configuration (['wearable'], etc.)."""
    names: list[str] = []
    if "wearable" in groups:
        names += wearable_feature_names()
    if "baseline" in groups:
        names += baseline_feature_names()
    if "clinical" in groups:
        names += clinical_feature_names()
    return names


def metadata_columns() -> list[str]:
    """Columns carried alongside features but never fed to a model."""
    return [f"{c}_hist_n" for c in config.WEARABLE_COLUMNS] + \
        ["rating_hist_n", "last_observed_date"]


def _features_for_patient(g: pd.DataFrame, with_baseline: bool) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    for col in config.WEARABLE_COLUMNS:
        s = g[col]
        lag1 = s.shift(1)  # yesterday's value — never today's
        g[f"{col}_lag1"] = lag1
        g[f"{col}_lag2"] = s.shift(2)
        g[f"{col}_roll7_mean"] = lag1.rolling(
            config.ROLLING_WINDOW,
            min_periods=config.ROLLING_MIN_PERIODS).mean()
        # shift-then-expand: the mean at row t uses only days < t
        hist = lag1.expanding(min_periods=config.EXPANDING_MIN_PERIODS).mean()
        g[f"{col}_hist_mean"] = hist
        g[f"{col}_dev_lag1"] = lag1 - hist
        g[f"{col}_lag1_missing"] = lag1.isna().astype(float)
        g[f"{col}_hist_n"] = lag1.expanding().count()
    if with_baseline and TARGET in g.columns:
        r = g[TARGET]
        lag1 = r.shift(1)  # yesterday's rating — never today's
        g["rating_lag1"] = lag1
        g["rating_lag2"] = r.shift(2)
        # shift-then-expand: mean of PAST ratings only (no leakage)
        g["rating_expanding_mean"] = lag1.expanding(min_periods=1).mean()
        prior_mean = r.shift(2).expanding(min_periods=1).mean()
        g["rating_dev_lag1"] = lag1 - prior_mean
        g["rating_lag1_missing"] = lag1.isna().astype(float)
        g["rating_hist_n"] = lag1.expanding().count()
    return g


def build_feature_frame(daily: pd.DataFrame, clinical: pd.DataFrame | None = None,
                        with_baseline: bool = True) -> pd.DataFrame:
    """Build the model-ready feature frame from the merged daily frame."""
    parts = []
    for pid, g in daily.groupby("patient_id", sort=True):
        parts.append(_features_for_patient(g, with_baseline))
    frame = pd.concat(parts, ignore_index=True)
    if clinical is not None:
        keep = ["patient_id"] + [c for c in CLINICAL_FEATURES if c in clinical.columns]
        frame = frame.merge(clinical[keep], on="patient_id", how="left")
    # "Last observed date" = last date with ANY recorded raw value. Placeholder
    # rows (next-day) carry no values, so they are never counted as observed.
    value_cols = [c for c in list(config.WEARABLE_COLUMNS) + [TARGET]
                  if c in frame.columns]
    has_obs = frame[value_cols].notna().any(axis=1)
    last_obs = (frame[has_obs]
                .groupby("patient_id")["date"].max()
                .rename("last_observed_date"))
    frame = frame.merge(last_obs, on="patient_id", how="left")
    return frame


def _extend_with_placeholders(daily: pd.DataFrame) -> pd.DataFrame:
    """Append one all-missing placeholder row (last_date + 1 day) per patient."""
    value_cols = list(config.WEARABLE_COLUMNS) + \
        ([TARGET] if TARGET in daily.columns else [])
    placeholders = []
    for pid, g in daily.groupby("patient_id", sort=True):
        row = {"patient_id": pid, "date": g["date"].max() + pd.Timedelta(days=1)}
        row.update({c: np.nan for c in value_cols})
        placeholders.append(row)
    extended = pd.concat([daily, pd.DataFrame(placeholders)], ignore_index=True)
    return extended.sort_values(["patient_id", "date"]).reset_index(drop=True)


def build_frames(daily: pd.DataFrame, clinical: pd.DataFrame | None = None,
                 with_baseline: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Observed-row feature frame + next-day rows from a single build pass."""
    full = build_feature_frame(_extend_with_placeholders(daily),
                               clinical=clinical, with_baseline=with_baseline)
    is_next = full["date"] > full["last_observed_date"]
    return full[~is_next].reset_index(drop=True), full[is_next].reset_index(drop=True)


def build_next_day_rows(daily: pd.DataFrame, clinical: pd.DataFrame | None = None,
                        with_baseline: bool = True) -> pd.DataFrame:
    """Feature rows for each patient's next (unobserved) day."""
    return build_frames(daily, clinical=clinical, with_baseline=with_baseline)[1]


def usable_for_training(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluation universe: rows with a known target AND at least one prior
    rating, so every predictor (including the persistence baseline) is scored
    on exactly the same rows."""
    ok = frame[TARGET].notna()
    if "rating_lag1" in frame.columns:
        ok &= frame["rating_lag1"].notna()
    return frame[ok].reset_index(drop=True)
