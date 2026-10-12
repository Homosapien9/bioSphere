"""Data validation gates for the Biosphere pipeline.

``validate_*`` functions inspect loaded frames and return a ValidationReport
with explicit errors (pipeline stops) and warnings (pipeline continues, UI
shows them). Nothing is ever repaired silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

import config


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"passed": self.passed, "errors": self.errors,
                "warnings": self.warnings}


class DataValidationError(Exception):
    """Raised by the pipeline when validation reports fatal errors."""


def _check_required_columns(df: pd.DataFrame, required: list[str],
                            label: str, report: ValidationReport) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        report.errors.append(
            f"{label}: missing required column(s): {', '.join(missing)}")


def _check_duplicates(df: pd.DataFrame, keys: list[str], label: str,
                      report: ValidationReport) -> None:
    if not all(k in df.columns for k in keys):
        return
    dup = df.duplicated(subset=keys).sum()
    if dup:
        report.errors.append(
            f"{label}: {int(dup)} duplicate ({', '.join(keys)}) row(s)")


def _check_date_continuity(df: pd.DataFrame, label: str,
                           report: ValidationReport) -> None:
    """Report (as warnings) gaps in the daily sequence per patient."""
    for pid, g in df.groupby("patient_id"):
        dates = g["date"].dropna().sort_values()
        if len(dates) < 2:
            continue
        full = pd.date_range(dates.min(), dates.max(), freq="D")
        gap = len(full) - dates.nunique()
        if gap > 0:
            report.warnings.append(
                f"{label}: patient {pid} has {gap} missing day(s) inside "
                f"{dates.min().date()}..{dates.max().date()}")


def validate_wearable(df: pd.DataFrame) -> ValidationReport:
    """Structural + plausibility checks on the daily wearable frame."""
    report = ValidationReport()
    required = ["patient_id", "date"] + config.WEARABLE_COLUMNS
    _check_required_columns(df, required, "wearable", report)
    _check_duplicates(df, ["patient_id", "date"], "wearable", report)

    if "date" in df.columns and df["date"].isna().any():
        report.errors.append(
            f"wearable: {int(df['date'].isna().sum())} unparseable date value(s)")

    for col, (low, high) in config.WEARABLE_RANGES.items():
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = int((vals.notna() & ~vals.between(low, high)).sum())
        if bad:
            report.warnings.append(
                f"wearable: {bad} value(s) of '{col}' outside plausible range "
                f"[{low}, {high}] (masked to missing at load time)")

    total = max(len(df), 1)
    for col in config.WEARABLE_COLUMNS:
        if col in df.columns:
            miss = df[col].isna().mean()
            if miss > 0.25:
                report.warnings.append(
                    f"wearable: column '{col}' is {miss:.0%} missing "
                    f"(>25% — check ingestion)")
    _check_date_continuity(df, "wearable", report)
    return report


def validate_diary(df: pd.DataFrame) -> ValidationReport:
    """Structural checks on the recovery-diary frame."""
    report = ValidationReport()
    required = ["patient_id", "date", config.RATING_COLUMN]
    _check_required_columns(df, required, "diary", report)
    _check_duplicates(df, ["patient_id", "date"], "diary", report)
    if config.RATING_COLUMN in df.columns:
        vals = pd.to_numeric(df[config.RATING_COLUMN], errors="coerce")
        bad = int((vals.notna() & ~vals.between(config.RATING_MIN, config.RATING_MAX)).sum())
        if bad:
            report.errors.append(
                f"diary: {bad} rating(s) outside the 1-5 scale")
    _check_date_continuity(df, "diary", report)
    return report


def validate_alignment(wearable: pd.DataFrame, diary: pd.DataFrame,
                       clinical: pd.DataFrame | None) -> ValidationReport:
    """Cross-stream checks: do the three views cover the same patients?"""
    report = ValidationReport()
    w_ids = set(wearable["patient_id"].unique())
    d_ids = set(diary["patient_id"].unique())
    c_ids = set(clinical["patient_id"].unique()) if clinical is not None else w_ids

    only_w = w_ids - d_ids
    only_d = d_ids - w_ids
    if only_w:
        report.warnings.append(
            f"alignment: {len(only_w)} patient(s) have wearable data but no diary "
            f"entries: {sorted(only_w)[:5]}")
    if only_d:
        report.warnings.append(
            f"alignment: {len(only_d)} patient(s) have diary entries but no wearable "
            f"data: {sorted(only_d)[:5]}")
    missing_clinical = w_ids - c_ids
    if clinical is not None and missing_clinical:
        report.warnings.append(
            f"alignment: {len(missing_clinical)} patient(s) lack a FHIR bundle: "
            f"{sorted(missing_clinical)[:5]}")

    if {"patient_id", "date"}.issubset(wearable.columns) and \
            {"patient_id", "date"}.issubset(diary.columns):
        w_span = wearable.groupby("patient_id")["date"].agg(["min", "max"])
        d_span = diary.groupby("patient_id")["date"].agg(["min", "max"])
        shared = w_span.index.intersection(d_span.index)
        disjoint = [p for p in shared
                    if w_span.loc[p, "max"] < d_span.loc[p, "min"]
                    or d_span.loc[p, "max"] < w_span.loc[p, "min"]]
        if disjoint:
            report.warnings.append(
                f"alignment: {len(disjoint)} patient(s) have non-overlapping "
                f"wearable/diary date ranges: {disjoint[:5]}")
    return report


def require_passed(report: ValidationReport, stage: str) -> None:
    """Raise a readable error if a report contains fatal errors."""
    if not report.passed:
        raise DataValidationError(
            f"{stage} validation failed: " + " | ".join(report.errors))
