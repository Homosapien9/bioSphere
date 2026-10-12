"""Pytest configuration: project root on sys.path + shared small-cohort fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import data_generation as DG  # noqa: E402
from src import fhir_loader as FL  # noqa: E402
from src import features as FE  # noqa: E402
from src import wearable_loader as WL  # noqa: E402

SMALL_SEED = 7
SMALL_PATIENTS = 6
SMALL_DAYS = 45


@pytest.fixture(scope="session")
def small_cohort():
    """Deterministic small cohort for fast tests."""
    return DG.generate_cohort(n_patients=SMALL_PATIENTS, n_days=SMALL_DAYS,
                              start_date="2026-07-01", seed=SMALL_SEED)


@pytest.fixture(scope="session")
def small_daily(small_cohort):
    """Wearable + diary merged into the model-ready daily frame."""
    wearable = small_cohort.wearable.copy()
    wearable["date"] = pd.to_datetime(wearable["date"])
    diary = small_cohort.diary.copy()
    diary["date"] = pd.to_datetime(diary["date"])
    return FE.merge_streams(wearable, diary)


@pytest.fixture(scope="session")
def small_clinical(small_cohort):
    profiles_all = [FL.parse_bundle(b) for b in small_cohort.bundles]
    return FL.build_clinical_features(profiles_all)


@pytest.fixture(scope="session")
def small_feature_frame(small_daily, small_clinical):
    return FE.build_feature_frame(small_daily, clinical=small_clinical)


@pytest.fixture(scope="session")
def loaded_demo(tmp_path_factory, small_cohort):
    """Small cohort written to disk and loaded back through the real loaders."""
    tmp = tmp_path_factory.mktemp("demo_data")
    DG.write_cohort(small_cohort, tmp)
    wearable, w_report = WL.load_wearable_csv(tmp / config.WEARABLE_CSV_NAME)
    diary, d_report = WL.load_diary_csv(tmp / config.DIARY_CSV_NAME)
    profiles = FL.load_patient_bundles(tmp / config.FHIR_DIR_NAME)
    return {"wearable": wearable, "diary": diary, "profiles": profiles,
            "w_report": w_report, "d_report": d_report, "dir": tmp}
