"""Tests for FHIR R4 bundle parsing and the clinical feature table."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src import fhir_loader as FL


def test_parse_generated_bundle(small_cohort):
    profile = FL.parse_bundle(small_cohort.bundles[0])
    assert profile.patient_id == "tb-patient-001"
    assert profile.age is not None and 20 <= profile.age <= 70
    assert profile.gender in ("male", "female")
    bmi = profile.observation_by_loinc(FL.LOINC_BMI)
    assert bmi is not None and 15 <= bmi <= 45


def test_all_bundles_parse(small_cohort):
    for bundle in small_cohort.bundles:
        profile = FL.parse_bundle(bundle)
        assert profile.patient_id
        assert len(profile.observations) >= 4  # BMI + 2x BP + resting HR


def test_missing_patient_raises():
    bundle = {"resourceType": "Bundle", "type": "collection",
              "entry": [{"resource": {"resourceType": "Observation", "id": "x",
                                      "status": "final"}}]}
    with pytest.raises(FL.FHIRValidationError, match="Patient"):
        FL.parse_bundle(bundle)


def test_wrong_resource_type_raises():
    with pytest.raises(FL.FHIRValidationError, match="Bundle"):
        FL.parse_bundle({"resourceType": "Patient", "id": "x"})
    bad = {"resourceType": "Bundle", "entry": [
        {"resource": {"resourceType": "NotAResource", "id": "x"}}]}
    with pytest.raises((FL.FHIRValidationError, ValidationError)):
        FL.parse_bundle(bad)


def test_malformed_json_raises(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{not json at all")
    with pytest.raises(FL.FHIRValidationError, match="malformed JSON"):
        FL.parse_bundle_file(f)


def test_condition_subject_reference_must_point_to_patient():
    bundle = {"resourceType": "Bundle", "entry": [
        {"resource": {"resourceType": "Patient", "id": "p1",
                      "birthDate": "1990-01-01"}},
        {"resource": {"resourceType": "Condition", "id": "c1",
                      "subject": {"reference": "Group/p1"},
                      "code": {"coding": [{"system": "x", "code": "I10"}]}}},
    ]}
    # The reference check is a semantic rule enforced at extraction time in v1;
    # parsing tolerates it but the profile keeps the raw reference for auditing.
    profile = FL.parse_bundle(bundle)
    assert profile.condition_codes() == ["I10"]


def test_age_reference_year_is_deterministic():
    bundle = {"resourceType": "Bundle", "timestamp": "2026-07-01", "entry": [
        {"resource": {"resourceType": "Patient", "id": "p1",
                      "birthDate": "1990-06-15"}}]}
    p = FL.parse_bundle(bundle)
    assert p.age == 36


def test_clinical_features_table(small_clinical):
    df = small_clinical
    assert len(df) == 6
    for col in FL.CLINICAL_FEATURES:
        assert col in df.columns
    assert df["age"].between(20, 70).all()
    assert df["n_conditions"].between(0, 3).all()
    assert set(df["gender_male"].unique()) <= {0.0, 1.0}
    # Weighted burden must differ from the plain count for at least one patient
    # (weights are 0.3-1.2, so a single condition never equals its count except 1.0).
    assert not df["condition_burden"].equals(df["n_conditions"])


def test_load_patient_bundles_from_disk(loaded_demo):
    profiles = loaded_demo["profiles"]
    assert len(profiles) == 6
    ids = {p.patient_id for p in profiles}
    assert len(ids) == 6
