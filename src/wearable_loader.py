"""Wearable CSV loading with type coercion and a transparent load report.

The loader accepts any CSV with the documented schema (patient_id, date, plus
the five daily wearable columns). Values that fail numeric parsing or fall
outside plausible ranges are converted to missing — and every such action is
recorded in the returned LoadReport, never applied silently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import config


@dataclass
class LoadReport:
    """What the loader did to the raw file — shown in the dashboard."""

    rows: int = 0
    patients: int = 0
    date_min: str | None = None
    date_max: str | None = None
    coerced_to_nan: dict = field(default_factory=dict)
    masked_implausible: dict = field(default_factory=dict)
    missing_after_load: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _coerce_frame(df: pd.DataFrame, value_columns: list[str],
                  report: LoadReport, mask_implausible: bool,
                  ranges: dict | None = None) -> pd.DataFrame:
    if "patient_id" not in df.columns:
        raise ValueError("CSV is missing the required 'patient_id' column")
    if "date" not in df.columns:
        raise ValueError("CSV is missing the required 'date' column")

    out = df.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    parsed_dates = pd.to_datetime(out["date"], errors="coerce")
    n_bad_dates = int(parsed_dates.isna().sum() - out["date"].isna().sum())
    if n_bad_dates:
        report.coerced_to_nan["date"] = n_bad_dates
    out["date"] = parsed_dates

    for col in value_columns:
        if col not in out.columns:
            # An entirely absent sensor column is allowed: it stays missing and
            # every downstream stage must report it, never invent values.
            out[col] = np.nan
            report.coerced_to_nan[f"{col} (column absent)"] = int(len(out))
            continue
        original = out[col]
        coerced = pd.to_numeric(original, errors="coerce")
        n_coerced = int(coerced.isna().sum() - original.isna().sum())
        if n_coerced:
            report.coerced_to_nan[col] = n_coerced
        if mask_implausible and ranges and col in ranges:
            low, high = ranges[col]
            bad = coerced.notna() & ~coerced.between(low, high)
            if bad.any():
                report.masked_implausible[col] = int(bad.sum())
                coerced[bad] = np.nan
        out[col] = coerced.astype(float)

    out = out.dropna(subset=["date"]).sort_values(["patient_id", "date"]).reset_index(drop=True)
    report.rows = int(len(out))
    report.patients = int(out["patient_id"].nunique())
    if len(out):
        report.date_min = out["date"].min().date().isoformat()
        report.date_max = out["date"].max().date().isoformat()
    for col in value_columns:
        if col in out.columns:
            report.missing_after_load[col] = int(out[col].isna().sum())
    return out


def load_wearable_csv(path, mask_implausible: bool = True) -> tuple[pd.DataFrame, LoadReport]:
    """Load the daily wearable CSV. Returns (dataframe, LoadReport)."""
    report = LoadReport()
    df = pd.read_csv(path)
    clean = _coerce_frame(df, config.WEARABLE_COLUMNS, report,
                          mask_implausible, config.WEARABLE_RANGES)
    return clean, report


def load_diary_csv(path) -> tuple[pd.DataFrame, LoadReport]:
    """Load the recovery-diary CSV (patient-reported 1-5 ratings)."""
    report = LoadReport()
    df = pd.read_csv(path)
    clean = _coerce_frame(df, [config.RATING_COLUMN], report, mask_implausible=False)
    # Ratings outside 1..5 (or non-integer) are labels, so they cannot be
    # silently masked: they are reported and dropped loudly instead.
    ratings = clean[config.RATING_COLUMN]
    out_of_scale = ratings.notna() & ~ratings.between(config.RATING_MIN, config.RATING_MAX)
    non_integer = ratings.notna() & (ratings % 1 != 0)
    for label, mask in [("out of 1-5", out_of_scale), ("non-integer", non_integer)]:
        if mask.any():
            report.masked_implausible[f"{config.RATING_COLUMN} ({label})"] = \
                int(mask.sum())
            clean.loc[mask, config.RATING_COLUMN] = np.nan
    return clean, report
