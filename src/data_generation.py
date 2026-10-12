"""Deterministic synthetic cohort generator for Biosphere.

Generates, for a seeded cohort of synthetic patients:
  1. Daily wearable records (sleep duration/timing, resting HR, HRV RMSSD, steps)
     with realistic dynamics, personal baselines, and honest missingness.
  2. A daily recovery diary: the self-reported 1-5 rating that is the prediction
     target. The label equation is documented in docs/dataset_card.md and is an
     ASSUMPTION BUILT INTO THE SIMULATOR, not a real medical finding.
  3. A valid FHIR R4 Bundle per patient (Patient, Observation, Condition,
     MedicationRequest resources) providing the structured clinical stream.

Everything is deterministic: the same seed reproduces byte-identical outputs.
No network access, no API keys, no real patient data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config

# ------------------------------------------------------------------------ --
# Clinical terminology libraries (illustrative codes; see dataset_card.md)
# ------------------------------------------------------------------------ --

CONDITION_LIBRARY = [
    {"code": "I10", "display": "Essential (primary) hypertension", "weight": 1.0,
     "rhr_up": 3.0, "hrv_down": 4.0, "recovery_down": 0.10},
    {"code": "E11.9", "display": "Type 2 diabetes mellitus without complications",
     "weight": 1.2, "rhr_up": 2.0, "hrv_down": 6.0, "recovery_down": 0.18},
    {"code": "J45.30", "display": "Mild persistent asthma, uncomplicated", "weight": 0.7,
     "rhr_up": 0.0, "hrv_down": 2.0, "recovery_down": 0.12},
    {"code": "K21.9", "display": "Gastro-esophageal reflux disease without esophagitis",
     "weight": 0.4, "rhr_up": 0.0, "hrv_down": 1.0, "recovery_down": 0.08},
    {"code": "M17.0", "display": "Unilateral primary osteoarthritis, knee", "weight": 0.6,
     "rhr_up": 0.0, "hrv_down": 0.0, "recovery_down": 0.10},
    {"code": "E55.9", "display": "Vitamin D deficiency, unspecified", "weight": 0.3,
     "rhr_up": 0.0, "hrv_down": 1.0, "recovery_down": 0.06},
    {"code": "M54.50", "display": "Low back pain, unspecified", "weight": 0.5,
     "rhr_up": 0.0, "hrv_down": 1.0, "recovery_down": 0.08},
]

# RxNorm ingredient-level codes (illustrative, ingredient level).
MEDICATION_LIBRARY = {
    "I10": {"code": "29046", "display": "Lisinopril (ingredient) oral"},
    "E11.9": {"code": "860975", "display": "Metformin (ingredient) oral"},
    "J45.30": {"code": "435", "display": "Albuterol (ingredient) inhalation"},
    "K21.9": {"code": "7626", "display": "Omeprazole (ingredient) oral"},
    "E55.9": {"code": "11256", "display": "Cholecalciferol (ingredient) oral"},
}
MED_CONDITION_PROB = 0.65       # probability an active condition has its routine med
MED_RECOVERY_EFFECT = 0.05      # per active medication, see dataset_card.md
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"
LOINC_SYSTEM = "http://loinc.org"
RXNORM_SYSTEM = "http://www.nlm.nih.gov/research/umls/rxnorm"

WEEKDAY_STEP_FACTOR = [1.05, 1.00, 0.98, 1.00, 1.04, 1.22, 1.12]  # Mon..Sun


@dataclass
class CohortData:
    """Container for one generated cohort."""
    wearable: pd.DataFrame
    diary: pd.DataFrame
    bundles: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------------ --
# FHIR bundle construction
# ------------------------------------------------------------------------ --

def _obs(resource_id: str, patient_id: str, loinc_code: str, display: str,
         value: float, unit: str, ucum: str, effective: str) -> dict:
    return {
        "resourceType": "Observation",
        "id": resource_id,
        "status": "final",
        "category": [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
            "code": "vital-signs"}]}],
        "code": {"coding": [{"system": LOINC_SYSTEM, "code": loinc_code,
                             "display": display}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": effective,
        "valueQuantity": {"value": round(float(value), 1), "unit": unit,
                          "system": "http://unitsofmeasure.org", "code": ucum},
    }


def build_fhir_bundle(patient_idx: int, traits: dict, conditions: list[dict],
                      meds: list[dict], start: pd.Timestamp) -> dict:
    """Build a valid FHIR R4 collection Bundle for one synthetic patient."""
    patient_id = traits["patient_id"]
    birth_year = traits["reference_year"] - int(traits["age"])
    birth_month = 1 + patient_idx % 9
    birth_day = 10 + patient_idx % 18
    birth_date = f"{birth_year:04d}-{birth_month:02d}-{birth_day:02d}"
    obs_date = (start - timedelta(days=30)).date().isoformat()

    entries: list[dict] = []

    patient_res = {
        "resourceType": "Patient",
        "id": patient_id,
        "gender": "male" if traits["gender_male"] else "female",
        "birthDate": birth_date,
    }
    entries.append({"fullUrl": f"urn:uuid:{patient_id}", "resource": patient_res})

    obs_specs = [
        ("bmi", "39156-5", "Body mass index (BMI) [Ratio]", traits["bmi"],
         "kg/m2", "kg/m2"),
        ("sysbp", "8480-6", "Systolic blood pressure", traits["systolic_bp"],
         "mm[Hg]", "mm[Hg]"),
        ("diabp", "8462-4", "Diastolic blood pressure", traits["diastolic_bp"],
         "mm[Hg]", "mm[Hg]"),
        ("resthr", "8867-4", "Heart rate", traits["clinical_resting_hr"],
         "beats/minute", "/min"),
    ]
    for key, loinc, disp, value, unit, ucum in obs_specs:
        entries.append({
            "fullUrl": f"urn:uuid:{patient_id}-{key}",
            "resource": _obs(f"{patient_id}-obs-{key}", patient_id, loinc, disp,
                             value, unit, ucum, obs_date),
        })

    for j, cond in enumerate(conditions):
        years_ago = 2 + (patient_idx + j) % 9
        onset = (start - timedelta(days=365 * years_ago)).date().isoformat()
        entries.append({
            "fullUrl": f"urn:uuid:{patient_id}-cond-{j}",
            "resource": {
                "resourceType": "Condition",
                "id": f"{patient_id}-cond-{j}",
                "clinicalStatus": {"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active"}]},
                "verificationStatus": {"coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed"}]},
                "code": {"coding": [{"system": ICD10_SYSTEM,
                                     "code": cond["code"],
                                     "display": cond["display"]}]},
                "subject": {"reference": f"Patient/{patient_id}"},
                "onsetDateTime": onset,
            },
        })

    for j, med in enumerate(meds):
        entries.append({
            "fullUrl": f"urn:uuid:{patient_id}-med-{j}",
            "resource": {
                "resourceType": "MedicationRequest",
                "id": f"{patient_id}-med-{j}",
                "status": "active",
                "intent": "order",
                "medicationCodeableConcept": {"coding": [{
                    "system": RXNORM_SYSTEM, "code": med["code"],
                    "display": med["display"]}]},
                "subject": {"reference": f"Patient/{patient_id}"},
                "authoredOn": obs_date,
            },
        })

    return {
        "resourceType": "Bundle",
        "id": f"bundle-{patient_id}",
        "type": "collection",
        "timestamp": start.date().isoformat(),
        "entry": entries,
    }


# ------------------------------------------------------------------------ --
# Cohort simulation
# ------------------------------------------------------------------------ --

def generate_cohort(n_patients: int = config.DEMO_N_PATIENTS,
                    n_days: int = config.DEMO_N_DAYS,
                    start_date: str = config.DEMO_START_DATE,
                    seed: int = config.RANDOM_SEED) -> CohortData:
    """Generate the full synthetic cohort deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(start_date)
    dates = pd.date_range(start, periods=n_days, freq="D")
    reference_year = start.year

    wearable_rows: list[dict] = []
    diary_rows: list[dict] = []
    bundles: list[dict] = []

    for p in range(n_patients):
        patient_id = f"tb-patient-{p + 1:03d}"

        # --- personal traits (fixed draw order per patient) ---------------
        age = int(rng.integers(25, 65))
        gender_male = bool(rng.random() < 0.5)
        sleep_need = float(rng.uniform(6.2, 8.8))
        chronotype = float(rng.normal(0.0, 0.9))       # + = night owl
        rhr_base = float(np.clip(rng.normal(62.0, 6.0), 48.0, 82.0))
        hrv_base = float(np.clip(rng.normal(58.0, 14.0), 25.0, 95.0))
        steps_base = float(rng.uniform(3500.0, 11500.0))
        base_recovery = float(np.clip(rng.normal(3.6, 0.45), 2.6, 4.6))
        bmi = float(np.clip(rng.normal(26.5, 3.8), 18.5, 38.0))

        # --- clinical draw (conditions + medications) ---------------------
        n_cond = int(rng.choice([0, 1, 2, 3], p=[0.30, 0.32, 0.23, 0.15]))
        cond_idx = rng.choice(len(CONDITION_LIBRARY), size=n_cond, replace=False)
        conditions = [CONDITION_LIBRARY[int(i)] for i in sorted(cond_idx)]
        cond_rhr = sum(c["rhr_up"] for c in conditions)
        cond_hrv = sum(c["hrv_down"] for c in conditions)
        cond_recovery = sum(c["recovery_down"] for c in conditions)
        meds = [MEDICATION_LIBRARY[c["code"]] for c in conditions
                if c["code"] in MEDICATION_LIBRARY and rng.random() < MED_CONDITION_PROB]
        n_meds = len(meds)

        systolic = float(np.clip(rng.normal(116 + 0.25 * (age - 40)
                                            + (4.0 if n_cond else 0.0), 8.0), 95, 160))
        diastolic = float(np.clip(rng.normal(74 + (3.0 if n_cond else 0.0), 6.0), 60, 98))
        clinical_rhr = round(rhr_base + cond_rhr)

        traits = {"patient_id": patient_id, "age": age, "gender_male": gender_male,
                  "sleep_need": sleep_need, "chronotype": chronotype,
                  "rhr_base": rhr_base, "hrv_base": hrv_base,
                  "steps_base": steps_base, "base_recovery": base_recovery,
                  "bmi": bmi, "systolic_bp": systolic, "diastolic_bp": diastolic,
                  "clinical_resting_hr": float(clinical_rhr),
                  "n_conditions": n_cond, "n_medications": n_meds,
                  "reference_year": reference_year}

        # --- optional device gap (consecutive days with no sensor data) ---
        has_gap = rng.random() < config.MISSING_DEVICE_GAP_PROB
        gap_len = int(rng.integers(3, 6)) if has_gap else 0
        gap_start = int(rng.integers(6, max(7, n_days - gap_len - 1))) if has_gap else -1

        # --- optional illness window --------------------------------------
        has_illness = rng.random() < 0.35
        ill_len = int(rng.integers(2, 5)) if has_illness else 0
        ill_start = int(rng.integers(10, max(11, n_days - ill_len - 2))) if has_illness else -1
        ill_rhr = float(rng.uniform(6.0, 11.0))
        ill_hrv = float(rng.uniform(12.0, 18.0))

        # --- daily dynamics ------------------------------------------------
        prev_sleep = sleep_need
        prev_latent = base_recovery
        for t in range(n_days):
            day = dates[t]
            dow = day.dayofweek

            short_sleep = rng.random() < 0.12
            long_sleep = (not short_sleep) and rng.random() < 0.06
            sleep = (sleep_need
                     + 0.45 * (prev_sleep - sleep_need)
                     + (0.35 if dow in (4, 5) else 0.0)
                     + (-1.9 if short_sleep else 0.0)
                     + (1.2 if long_sleep else 0.0)
                     + rng.normal(0.0, 0.55))
            onset = (22.6 + chronotype * 0.9
                     + (1.3 if short_sleep else 0.0)
                     + rng.normal(0.0, 0.7))
            onset = float(np.clip(onset, 20.0, 27.5))

            # Sleep debt: tonight's shortfall against this person's sleep need.
            # (A 3-night trailing average was considered; tonight's shortfall
            # dominates the simulated physiology — see dataset_card.md.)
            debt3 = max(0.0, sleep_need - sleep)

            ill_flag = 1.0 if (has_illness and ill_start <= t < ill_start + ill_len) else 0.0
            resting_hr = rhr_base + cond_rhr + 2.2 * debt3 + ill_flag * ill_rhr \
                + rng.normal(0.0, 2.0)
            hrv = hrv_base - cond_hrv - 2.4 * debt3 - ill_flag * ill_hrv \
                + rng.normal(0.0, 4.0)
            steps = steps_base * WEEKDAY_STEP_FACTOR[dow] * (1.0 - 0.5 * ill_flag) \
                * float(np.exp(rng.normal(0.0, 0.22)))

            # ---------------- label equation (documented assumption) ------
            latent = (base_recovery
                      + 0.45 * (sleep - sleep_need)
                      + 0.30 * (hrv - hrv_base) / 8.0
                      - 0.25 * (resting_hr - rhr_base) / 4.0
                      + 0.20 * (steps - steps_base) / 4000.0
                      - 0.10 * (onset - (22.6 + chronotype * 0.9))
                      - cond_recovery
                      - 0.16 * min(n_cond, 3)
                      - MED_RECOVERY_EFFECT * n_meds
                      - 1.10 * ill_flag
                      + 0.22 * (prev_latent - 3.0)
                      + rng.normal(0.0, 0.45))
            rating = float(np.clip(round(latent), 1, 5))
            prev_latent = latent
            prev_sleep = sleep

            wearable_rows.append({
                "patient_id": patient_id,
                "date": day.strftime("%Y-%m-%d"),
                "sleep_duration_hours": round(sleep, 2),
                "sleep_onset_hour": round(onset, 2),
                "resting_hr": round(resting_hr, 1),
                "hrv_rmssd": round(hrv, 1),
                "steps": round(steps, 0),
            })
            # Diary entries are missing when the patient forgets to log (p=5%).
            reported = rng.random() >= config.MISSING_DIARY_PROB
            diary_rows.append({
                "patient_id": patient_id,
                "date": day.strftime("%Y-%m-%d"),
                "recovery_rating": rating if reported else np.nan,
            })

        # Inject the device gap (all sensors absent, diary may still exist).
        # The last n_days appended rows are exactly day offsets 0..n_days-1.
        if has_gap:
            for offset, row in enumerate(wearable_rows[-n_days:]):
                if gap_start <= offset < gap_start + gap_len:
                    for col in config.WEARABLE_COLUMNS:
                        row[col] = np.nan

        # Inject sparse per-cell missingness (sensor dropouts).
        for row in wearable_rows[-n_days:]:
            for col in config.WEARABLE_COLUMNS:
                if rng.random() < config.MISSING_SENSOR_CELL_PROB:
                    row[col] = np.nan

        bundles.append(build_fhir_bundle(p, traits, conditions, meds, start))

    wearable = pd.DataFrame(wearable_rows)
    diary = pd.DataFrame(diary_rows)
    return CohortData(wearable=wearable, diary=diary, bundles=bundles)


# ------------------------------------------------------------------------ --
# Persistence helpers
# ------------------------------------------------------------------------ --

def write_cohort(cohort: CohortData, out_dir: Path) -> dict:
    """Write wearable CSV, diary CSV, and per-patient FHIR bundles to ``out_dir``."""
    out_dir = Path(out_dir)
    (out_dir / config.FHIR_DIR_NAME).mkdir(parents=True, exist_ok=True)

    wearable_path = out_dir / config.WEARABLE_CSV_NAME
    diary_path = out_dir / config.DIARY_CSV_NAME
    cohort.wearable.to_csv(wearable_path, index=False)
    cohort.diary.to_csv(diary_path, index=False)

    fhir_paths = []
    for bundle in cohort.bundles:
        fpath = out_dir / config.FHIR_DIR_NAME / f"{bundle['id']}.json"
        fpath.write_text(json.dumps(bundle, indent=1))
        fhir_paths.append(fpath.name)

    return {"wearable_csv": str(wearable_path), "diary_csv": str(diary_path),
            "fhir_dir": str(out_dir / config.FHIR_DIR_NAME),
            "n_bundles": len(fhir_paths)}
