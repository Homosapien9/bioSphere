"""Personal physiological baselines (display & evidence layer).

For each patient and each daily measurement, maintains a robust historical
reference (mean, median, MAD-based dispersion) computed ONLY from the history
supplied at fit time, plus a robust z-score comparing the most recent days
against that reference.

Important boundaries:
- A statistical deviation is NOT a disease, a diagnosis, or a proven
  adverse-event signal. It is a description of where recent values sit
  relative to this person's own history.
- A baseline with few supporting observations is flagged insufficient rather
  than presented as stable.
- Missing signals are reported as unavailable; nothing is invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

_MAD_SCALE = 1.4826  # consistency constant: MAD -> sigma under normality


class PersonalBaseline:
    """Per-patient, per-metric robust baselines over the provided history."""

    def __init__(self, min_obs: int = config.BASELINE_MIN_OBS,
                 metrics: list[str] | None = None):
        self.min_obs = min_obs
        self.metrics = metrics or (config.WEARABLE_COLUMNS + [config.RATING_COLUMN])
        self._stats: dict[tuple[str, str], dict] = {}

    def fit(self, daily: pd.DataFrame) -> "PersonalBaseline":
        """Compute baselines from the given history (caller controls the cutoff)."""
        self._stats.clear()
        has_rating = config.RATING_COLUMN in daily.columns
        for pid, g in daily.groupby("patient_id", sort=True):
            for metric in self.metrics:
                if metric == config.RATING_COLUMN and not has_rating:
                    continue
                stats: dict = {"patient_id": pid, "metric": metric}
                if metric not in g.columns:
                    stats.update(n_obs=0, sufficient=False, status="unavailable",
                                 reason="column absent from the loaded data")
                    self._stats[(pid, metric)] = stats
                    continue
                values = pd.to_numeric(g[metric], errors="coerce").dropna()
                stats.update(n_obs=int(len(values)),
                             sufficient=bool(len(values) >= self.min_obs))
                if values.empty:
                    stats.update(status="unavailable",
                                 reason="no observations recorded for this metric")
                else:
                    arr = values.to_numpy(dtype=float)
                    median = float(np.median(arr))
                    mad = float(np.median(np.abs(arr - median)))
                    stats.update(
                        status="ok",
                        mean=round(float(arr.mean()), 3),
                        median=round(median, 3),
                        std=round(float(arr.std(ddof=1)), 3) if len(arr) > 1 else None,
                        # Zero dispersion is a real measurement, not an absence —
                        # report it as 0.0; robust z-scores become unavailable.
                        mad_sigma=round(mad * _MAD_SCALE, 3))
                self._stats[(pid, metric)] = stats
        return self

    def summary(self, patient_id: str, metric: str) -> dict | None:
        """Baseline summary for one patient/metric (None if unknown metric)."""
        return self._stats.get((patient_id, metric))

    def recent_comparison(self, daily: pd.DataFrame, patient_id: str, metric: str,
                          recent_k: int = config.RECENT_WINDOW_DAYS) -> dict:
        """Compare the most recent ``recent_k`` observations with the baseline.

        Returns a dict with the recent mean, the baseline median, a robust
        z-score when dispersion is estimable, and explicit availability flags.
        """
        stats = self.summary(patient_id, metric)
        if stats is None or stats.get("status") != "ok":
            reason = stats.get("reason") if stats else "metric unknown"
            return {"status": "unavailable", "reason": reason, "metric": metric}

        g = daily[daily["patient_id"] == patient_id].sort_values("date")
        recent = pd.to_numeric(g[metric], errors="coerce").dropna().tail(recent_k)
        if recent.empty:
            return {"status": "unavailable", "reason": "no recent observations",
                    "metric": metric, "n_baseline": stats["n_obs"]}

        out = {
            "status": "ok",
            "metric": metric,
            "n_baseline": stats["n_obs"],
            "sufficient": stats["sufficient"],
            "recent_k": int(len(recent)),
            "recent_mean": round(float(recent.mean()), 3),
            "baseline_median": stats["median"],
            "baseline_mean": stats["mean"],
        }
        if stats["mad_sigma"] and stats["mad_sigma"] > 0:
            out["robust_z"] = round(
                (out["recent_mean"] - stats["median"]) / stats["mad_sigma"], 2)
        else:
            out["robust_z"] = None
            out["note"] = "dispersion is zero or too small for a robust z-score"
        if not stats["sufficient"]:
            out["note"] = (
                f"insufficient history: only {stats['n_obs']} observations "
                f"(< {self.min_obs}); treat this baseline as provisional")
        return out

    def completeness(self, daily: pd.DataFrame, patient_id: str) -> dict:
        """Per-metric data completeness for the dashboard overview."""
        g = daily[daily["patient_id"] == patient_id]
        out = {}
        for metric in self.metrics:
            if metric not in g.columns:
                continue
            total = len(g)
            present = int(pd.to_numeric(g[metric], errors="coerce").notna().sum())
            s = self.summary(patient_id, metric) or {}
            out[metric] = {
                "present": present, "total": total,
                "pct": round(100.0 * present / total, 1) if total else 0.0,
                "sufficient_history": s.get("sufficient", False),
            }
        return out
