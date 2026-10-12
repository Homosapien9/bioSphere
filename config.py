"""Biosphere — central configuration.

Every tunable constant used by the pipeline lives here so that tests, scripts,
and the dashboard all agree on one set of values. No environment variables and
no network access are required anywhere in this project.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEMO_DATA_DIR = DATA_DIR / "demo"
GENERATED_DATA_DIR = DATA_DIR / "generated"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

WEARABLE_CSV_NAME = "wearable_daily.csv"
DIARY_CSV_NAME = "recovery_diary.csv"
FHIR_DIR_NAME = "fhir"

# ------------------------------------------------------------- task ------
# Prediction target: the patient's self-reported recovery rating for day D+1,
# on an integer 1-5 scale (1 = very poorly recovered, 5 = very well recovered).
# Forecast horizon: exactly 1 day ahead. Features may only use information
# recorded strictly before day D+1 (i.e., wearable data and diary entries
# through day D).
RATING_COLUMN = "recovery_rating"
RATING_MIN, RATING_MAX = 1.0, 5.0
FORECAST_HORIZON_DAYS = 1

# Wearable measurement columns (daily aggregates).
WEARABLE_COLUMNS = [
    "sleep_duration_hours",
    "sleep_onset_hour",
    "resting_hr",
    "hrv_rmssd",
    "steps",
]

# Units and plausibility ranges used by validation (daily aggregates).
WEARABLE_RANGES = {
    "sleep_duration_hours": (0.0, 16.0),
    "sleep_onset_hour": (18.0, 30.0),  # hours on a 24h clock; 24+ = after midnight
    "resting_hr": (30.0, 140.0),
    "hrv_rmssd": (2.0, 250.0),
    "steps": (0.0, 60_000.0),
}

# ------------------------------------------------------------- data gen ----
RANDOM_SEED = 20260214
DEMO_N_PATIENTS = 24
DEMO_N_DAYS = 90
DEMO_START_DATE = "2026-07-01"
MISSING_SENSOR_CELL_PROB = 0.06   # per-cell probability a wearable value is absent
MISSING_DEVICE_GAP_PROB = 0.18    # per-patient probability of a 3-5 day device gap
MISSING_DIARY_PROB = 0.05         # probability a daily diary entry is missing

# ------------------------------------------------------- evaluation split --
TEST_FRACTION = 0.25               # last 25% of each patient's usable days -> test
MIN_TRAIN_ROWS_PER_PATIENT = 10    # patients with fewer usable days are train-only
HELDOUT_PATIENT_MODULUS = 5        # every 5th patient (sorted id) is held out entirely

# ---------------------------------------------------------- features -------
ROLLING_WINDOW = 7
ROLLING_MIN_PERIODS = 3
EXPANDING_MIN_PERIODS = 3

# Feature groups (cumulative configurations for the ablation study):
#   wearable_only  : lagged/rolled wearable signals only
#   with_baseline  : + recovery-rating history (persistence-style personal features)
#   with_clinical  : + FHIR-derived static clinical context
CONFIGURATIONS = {
    "wearable_only": ["wearable"],
    "with_baseline": ["wearable", "baseline"],
    "with_clinical": ["wearable", "baseline", "clinical"],
}

# ---------------------------------------------------------- models ---------
RIDGE_ALPHA = 1.0

# --------------------------------------------------- personal baseline -----
BASELINE_MIN_OBS = 7               # observations needed before a baseline is "stable"
RECENT_WINDOW_DAYS = 3             # "recent" window compared against the baseline

# ---------------------------------------------------------- scenarios ------
SCENARIO_SLEEP_RANGE = (3.0, 12.0)
SCENARIO_TRAIN_RANGE_PERCENTILES = (1.0, 99.0)

# ---------------------------------------------------------- metadata -------
PROJECT_NAME = "Biosphere"
PROJECT_TAGLINE = "Your physiology, understood over time."
PROJECT_VERSION = "0.2.0"
SYNTHETIC_DISCLAIMER = (
    "All data in this demo are synthetic (generated with a seeded simulator) and all "
    "predictions are model outputs from a research prototype. Biosphere is not a "
    "medical device and nothing here is medical advice, a diagnosis, or a causal estimate."
)
