"""Single ingestion path shared by the dashboard, scripts, and smoke test.

Centralizing loading guarantees the dashboard, the evaluation script, and the
smoke test all consume identical inputs: same loaders, same reference year,
same parsing. Every load action is recorded in the returned LoadReports.
"""
from __future__ import annotations

from pathlib import Path

import config
from src import fhir_loader as FL
from src import features as FE
from src import wearable_loader as WL


def load_inputs(data_dir) -> dict:
    """Load the three data streams and build the merged daily frame."""
    data_dir = Path(data_dir)
    wearable, w_report = WL.load_wearable_csv(data_dir / config.WEARABLE_CSV_NAME)
    diary, d_report = WL.load_diary_csv(data_dir / config.DIARY_CSV_NAME)
    profiles = FL.load_patient_bundles(
        data_dir / config.FHIR_DIR_NAME,
        reference_year=int(config.DEMO_START_DATE[:4]))
    clinical = FL.build_clinical_features(profiles)
    return {
        "dir": str(data_dir),
        "wearable": wearable, "w_report": w_report,
        "diary": diary, "d_report": d_report,
        "profiles": profiles, "clinical": clinical,
        "daily": FE.merge_streams(wearable, diary),
    }
