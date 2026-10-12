"""Tests for the deterministic synthetic data generator."""
from __future__ import annotations

import pandas as pd

import config
from src import data_generation as DG


def test_generation_is_deterministic():
    a = DG.generate_cohort(n_patients=3, n_days=20, start_date="2026-07-01", seed=123)
    b = DG.generate_cohort(n_patients=3, n_days=20, start_date="2026-07-01", seed=123)
    pd.testing.assert_frame_equal(a.wearable, b.wearable)
    pd.testing.assert_frame_equal(a.diary, b.diary)
    assert a.bundles[0] == b.bundles[0]


def test_different_seed_gives_different_data():
    a = DG.generate_cohort(n_patients=3, n_days=20, start_date="2026-07-01", seed=1)
    b = DG.generate_cohort(n_patients=3, n_days=20, start_date="2026-07-01", seed=2)
    assert not a.wearable.equals(b.wearable)


def test_wearable_schema_and_ranges(small_cohort):
    w = small_cohort.wearable
    assert list(w["patient_id"].unique()) == sorted(w["patient_id"].unique())
    for col, (low, high) in config.WEARABLE_RANGES.items():
        vals = w[col].dropna()
        assert vals.between(low, high).all(), f"{col} out of range"


def test_ratings_are_whole_numbers_in_1_to_5(small_cohort):
    r = small_cohort.diary[config.RATING_COLUMN].dropna()
    assert len(r) > 0
    assert r.between(1, 5).all()
    assert (r % 1 == 0).all()


def test_missingness_is_present_but_not_extreme(small_cohort):
    w = small_cohort.wearable
    # Sparse cell dropout is planted: some values must be missing.
    assert w["sleep_duration_hours"].isna().sum() > 0
    # ... but not so much that the columns are mostly empty.
    for col in config.WEARABLE_COLUMNS:
        assert w[col].isna().mean() < 0.25
    d = small_cohort.diary[config.RATING_COLUMN]
    assert d.isna().sum() > 0 and d.isna().mean() < 0.25


def test_diary_covers_every_patient_day(small_cohort):
    d = small_cohort.diary
    assert len(d) == len(small_cohort.wearable)
    for pid, g in d.groupby("patient_id"):
        dates = pd.to_datetime(g["date"])
        assert (dates.diff().dropna() == pd.Timedelta(days=1)).all()


def test_fhir_bundle_structure(small_cohort):
    b = small_cohort.bundles[0]
    assert b["resourceType"] == "Bundle"
    assert b["type"] == "collection"
    resources = [e["resource"] for e in b["entry"]]
    types = [r["resourceType"] for r in resources]
    assert types.count("Patient") == 1
    assert "Observation" in types
    for r in resources:
        if r["resourceType"] in ("Condition", "Observation", "MedicationRequest"):
            assert r["subject"]["reference"].startswith("Patient/")
    # Every LOINC observation carries a value quantity.
    for r in resources:
        if r["resourceType"] == "Observation":
            assert "valueQuantity" in r


def test_write_cohort_is_byte_stable(tmp_path, small_cohort):
    """Writing the same cohort twice must produce byte-identical files:
    the shipped demo dataset and reproducibility depend on this."""
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    DG.write_cohort(small_cohort, out_a)
    DG.write_cohort(small_cohort, out_b)
    for rel in (config.WEARABLE_CSV_NAME, config.DIARY_CSV_NAME):
        assert (out_a / rel).read_bytes() == (out_b / rel).read_bytes(), rel
    names_a = sorted(p.name for p in (out_a / config.FHIR_DIR_NAME).glob("*.json"))
    names_b = sorted(p.name for p in (out_b / config.FHIR_DIR_NAME).glob("*.json"))
    assert names_a == names_b and names_a
    for name in names_a:
        assert (out_a / config.FHIR_DIR_NAME / name).read_bytes() == \
            (out_b / config.FHIR_DIR_NAME / name).read_bytes(), name
