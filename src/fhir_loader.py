"""FHIR R4 bundle loading with Pydantic validation.

Parses local FHIR R4 collection Bundles (one JSON file per synthetic patient)
into typed profiles, then derives the static clinical-context feature table.
Pattern adapted conceptually from PatientPulse (MIT), re-implemented for local
files; see LICENSES_AND_ATTRIBUTION.md.

Validation is strict where it matters: a bundle must have resourceType
"Bundle", contain exactly one Patient resource, and every resource we consume
must carry the expected resourceType. Unknown extra fields are allowed (real
FHIR resources carry many fields we do not use).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class FHIRValidationError(Exception):
    """Raised when a FHIR bundle cannot be parsed or fails schema checks."""


# --------------------------------------------------------------- resources --

class _FHIRBase(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Coding(_FHIRBase):
    system: str | None = None
    code: str | None = None
    display: str | None = None


class CodeableConcept(_FHIRBase):
    coding: list[Coding] = Field(default_factory=list)
    text: str | None = None

    def first_code(self) -> Coding | None:
        return self.coding[0] if self.coding else None


class Reference(_FHIRBase):
    reference: str | None = None
    display: str | None = None


class Quantity(_FHIRBase):
    value: float | None = None
    unit: str | None = None
    code: str | None = None


class FHIRPatient(_FHIRBase):
    resource_type: str = Field(alias="resourceType")
    id: str
    gender: str | None = None
    birth_date: str | None = Field(alias="birthDate", default=None)

    @field_validator("resource_type")
    @classmethod
    def _rt(cls, v: str) -> str:
        if v != "Patient":
            raise ValueError(f"expected resourceType 'Patient', got '{v}'")
        return v


class FHIRCondition(_FHIRBase):
    resource_type: str = Field(alias="resourceType")
    id: str
    code: CodeableConcept | None = None
    subject: Reference | None = None
    onset_date_time: str | None = Field(alias="onsetDateTime", default=None)

    @field_validator("resource_type")
    @classmethod
    def _rt(cls, v: str) -> str:
        if v != "Condition":
            raise ValueError(f"expected resourceType 'Condition', got '{v}'")
        return v


class FHIRMedicationRequest(_FHIRBase):
    resource_type: str = Field(alias="resourceType")
    id: str
    status: str | None = None
    medication_codeable_concept: CodeableConcept | None = \
        Field(alias="medicationCodeableConcept", default=None)
    subject: Reference | None = None

    @field_validator("resource_type")
    @classmethod
    def _rt(cls, v: str) -> str:
        if v != "MedicationRequest":
            raise ValueError(f"expected resourceType 'MedicationRequest', got '{v}'")
        return v


class FHIRObservation(_FHIRBase):
    resource_type: str = Field(alias="resourceType")
    id: str
    status: str | None = None
    code: CodeableConcept | None = None
    subject: Reference | None = None
    effective_date_time: str | None = Field(alias="effectiveDateTime", default=None)
    value_quantity: Quantity | None = Field(alias="valueQuantity", default=None)

    @field_validator("resource_type")
    @classmethod
    def _rt(cls, v: str) -> str:
        if v != "Observation":
            raise ValueError(f"expected resourceType 'Observation', got '{v}'")
        return v


class PatientClinicalProfile(BaseModel):
    """Typed, validated view of one patient's clinical bundle."""

    model_config = ConfigDict(extra="allow")

    patient_id: str
    gender: str | None
    birth_date: str | None
    age: int | None
    observations: list[FHIRObservation]
    conditions: list[FHIRCondition]
    medications: list[FHIRMedicationRequest]
    source_file: str | None = None

    def observation_by_loinc(self, loinc_code: str) -> float | None:
        """Most recent valueQuantity for the given LOINC code, if present."""
        best = None
        for obs in self.observations:
            code = obs.code.first_code() if obs.code else None
            if not code or code.code != loinc_code:
                continue
            if obs.value_quantity is None or obs.value_quantity.value is None:
                continue
            best = obs  # entries are stored in generation order; keep the last
        return best.value_quantity.value if best else None

    def condition_codes(self) -> list[str]:
        out = []
        for cond in self.conditions:
            code = cond.code.first_code() if cond.code else None
            if code and code.code:
                out.append(code.code)
        return out

    def medication_codes(self) -> list[str]:
        out = []
        for med in self.medications:
            code = med.medication_codeable_concept.first_code() \
                if med.medication_codeable_concept else None
            if code and code.code:
                out.append(code.code)
        return out


# ------------------------------------------------------------------ parsing --

def _parse_age(birth_date: str | None, reference_year: int) -> int | None:
    if not birth_date:
        return None
    try:
        born = date.fromisoformat(birth_date)
    except ValueError as exc:
        raise FHIRValidationError(f"unparseable Patient.birthDate '{birth_date}': {exc}") \
            from exc
    age = reference_year - born.year
    if age < 0 or age > 120:
        raise FHIRValidationError(f"implausible age {age} from birthDate '{birth_date}'")
    return age


def parse_bundle(bundle: dict, source_file: str | None = None,
                 reference_year: int | None = None) -> PatientClinicalProfile:
    """Validate one parsed FHIR Bundle dict and extract a clinical profile."""
    if not isinstance(bundle, dict):
        raise FHIRValidationError("bundle must be a JSON object")
    rtype = bundle.get("resourceType")
    if rtype != "Bundle":
        raise FHIRValidationError(f"expected resourceType 'Bundle', got '{rtype!r}'")

    entries = bundle.get("entry") or []
    resources = [e.get("resource", {}) for e in entries if isinstance(e, dict)]

    patients = [r for r in resources if r.get("resourceType") == "Patient"]
    if len(patients) != 1:
        raise FHIRValidationError(
            f"bundle must contain exactly one Patient resource, found {len(patients)}")

    if reference_year is None:
        ts = bundle.get("timestamp")
        try:
            reference_year = int(str(ts)[:4]) if ts else date.today().year
        except (TypeError, ValueError):
            reference_year = date.today().year

    try:
        patient = FHIRPatient.model_validate(patients[0])
    except ValidationError as exc:
        raise FHIRValidationError(f"invalid Patient resource: {exc}") from exc

    conditions, medications, observations = [], [], []
    handlers = {"Condition": (FHIRCondition, conditions),
                "MedicationRequest": (FHIRMedicationRequest, medications),
                "Observation": (FHIRObservation, observations)}
    for i, res in enumerate(resources):
        rt = res.get("resourceType")
        if rt not in handlers:
            continue  # Patient handled above; other resource types tolerated, unused
        model, bucket = handlers[rt]
        try:
            bucket.append(model.model_validate(res))
        except ValidationError as exc:
            raise FHIRValidationError(
                f"invalid {rt} resource at entry {i} of {source_file or 'bundle'}: {exc}"
            ) from exc

    return PatientClinicalProfile(
        patient_id=patient.id,
        gender=patient.gender,
        birth_date=patient.birth_date,
        age=_parse_age(patient.birth_date, reference_year),
        observations=observations,
        conditions=conditions,
        medications=medications,
        source_file=source_file,
    )


def parse_bundle_file(path, reference_year: int | None = None) -> PatientClinicalProfile:
    """Load and validate one FHIR bundle JSON file."""
    path = Path(path)
    try:
        bundle = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise FHIRValidationError(f"{path.name}: malformed JSON ({exc})") from exc
    return parse_bundle(bundle, source_file=path.name, reference_year=reference_year)


def load_patient_bundles(fhir_dir, reference_year: int | None = None) \
        -> list[PatientClinicalProfile]:
    """Load every *.json bundle in ``fhir_dir`` (sorted for determinism)."""
    fhir_dir = Path(fhir_dir)
    if not fhir_dir.is_dir():
        raise FHIRValidationError(f"FHIR directory not found: {fhir_dir}")
    files = sorted(fhir_dir.glob("*.json"))
    if not files:
        raise FHIRValidationError(f"no FHIR bundle JSON files in {fhir_dir}")
    profiles = []
    problems = []
    for f in files:
        try:
            profiles.append(parse_bundle_file(f, reference_year=reference_year))
        except FHIRValidationError as exc:
            problems.append(str(exc))
    if problems:
        shown = " | ".join(problems[:3])
        more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
        raise FHIRValidationError(
            f"FHIR validation failed for {len(problems)} bundle(s): {shown}{more}")
    return profiles


# ------------------------------------------------------------- feature table --

# LOINC codes consumed for clinical-context features.
LOINC_BMI = "39156-5"
LOINC_SYSTOLIC = "8480-6"
LOINC_DIASTOLIC = "8462-4"
LOINC_RESTING_HR = "8867-4"

CONDITION_FLAGS = {
    "has_hypertension": "I10",
    "has_diabetes": "E11.9",
    "has_asthma": "J45.30",
}

# Documented severity weights (same values as the generator's library; see
# docs/dataset_card.md). Used to build a weighted condition burden instead of
# a plain count, avoiding an exactly duplicated feature column.
CONDITION_WEIGHTS = {
    "I10": 1.0, "E11.9": 1.2, "J45.30": 0.7, "K21.9": 0.4,
    "M17.0": 0.6, "E55.9": 0.3, "M54.50": 0.5,
}

CLINICAL_FEATURES = [
    "age", "gender_male", "bmi", "systolic_bp", "diastolic_bp",
    "clinical_resting_hr", "n_conditions", "n_medications",
    "has_hypertension", "has_diabetes", "has_asthma", "condition_burden",
]


def build_clinical_features(profiles: list[PatientClinicalProfile]) -> pd.DataFrame:
    """One row per patient with interpretable FHIR-derived context features."""
    rows = []
    for p in profiles:
        codes = p.condition_codes()
        rows.append({
            "patient_id": p.patient_id,
            "age": p.age,
            "gender_male": 1.0 if (p.gender or "").lower() == "male" else 0.0,
            "bmi": p.observation_by_loinc(LOINC_BMI),
            "systolic_bp": p.observation_by_loinc(LOINC_SYSTOLIC),
            "diastolic_bp": p.observation_by_loinc(LOINC_DIASTOLIC),
            "clinical_resting_hr": p.observation_by_loinc(LOINC_RESTING_HR),
            "n_conditions": float(len(codes)),
            "n_medications": float(len(p.medication_codes())),
            "has_hypertension": 1.0 if CONDITION_FLAGS["has_hypertension"] in codes else 0.0,
            "has_diabetes": 1.0 if CONDITION_FLAGS["has_diabetes"] in codes else 0.0,
            "has_asthma": 1.0 if CONDITION_FLAGS["has_asthma"] in codes else 0.0,
            "condition_burden": float(min(
                sum(CONDITION_WEIGHTS.get(c, 0.5) for c in codes), 3.0)),
            "_condition_codes": codes,
            "_medication_codes": p.medication_codes(),
        })
    return pd.DataFrame(rows)
