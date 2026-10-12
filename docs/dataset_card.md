# Biosphere — Dataset Card (v1, synthetic)

**Status:** all data in this project are synthetic, generated deterministically by
`src/data_generation.py` with seed `20260214` (configurable). **No real patient data
exists anywhere in this repository.** Nothing in this card — or in any result derived
from this data — constitutes evidence about real human physiology, and no performance
number measured on these labels may be presented as medical validity.

---

## 1. Why synthetic (verified dataset investigation)

The brief requires one wearable stream and one structured clinical (EHR/FHIR) stream
for the *same* person, plus a daily self-reported outcome. We verified the three
suggested sources directly during the Phase 1 audit:

| Source | License / access (verified) | Why it is not used in v1 |
|---|---|---|
| **Synthea** (GitHub `synthetichealth/synthea`) | Apache-2.0 (verified from raw `LICENSE`) | Generates clinical records only — no wearable time series and no daily self-rated recovery outcome. (Its *concept* of linked synthetic patients inspired our generator.) |
| **WESAD** (UCI `dataset/465/wesad`) | UCI page verified: 15 subjects, lab stress-affect study, physiological + motion sensors; license deferred to a linked dataset page | Hourly-resolution lab sessions, not daily home monitoring; no EHR pairing; no daily 1–5 recovery self-ratings; license requires following a secondary link. |
| **BIDSleep** (PhysioNet `content/bidsleep/1.0.1`) | **HTTP 404 at audit time** (both the versioned and bare URLs) — could not be verified at all | Unverifiable availability; PhysioNet-hosted sleep datasets of this kind are also tens of GB and require the PhysioNet credentialed-access flow. |

**Decision:** per the brief's explicit fallback provision, v1 ships a deterministic
synthetic generator. The wearable loader accepts any CSV with the same schema, so a
real export (e.g., a future Apple-Health-derived daily table) can be plugged in
without touching the pipeline; any such experiment must be reported separately from
synthetic results.

## 2. Cohort

- **24 patients × 90 days** (2026-07-01 → 2026-09-28), 2,160 daily records.
- Per-patient latent traits (drawn once): age 25–64, sex (50/50), sleep need
  U(6.2, 8.8) h, chronotype N(0, 0.9) h, resting-HR base N(62, 6) clipped [48, 82],
  HRV base N(58, 14) clipped [25, 95] ms, activity base U(3,500, 11,500) steps,
  baseline recovery N(3.6, 0.45) clipped [2.6, 4.6], BMI N(26.5, 3.8) clipped
  [18.5, 38].
- Clinical draw: 0–3 conditions from a 7-condition ICD-10-CM library (hypertension
  I10, T2DM E11.9, mild persistent asthma J45.30, GERD K21.9, knee osteoarthritis
  M17.0, vitamin-D deficiency E55.9, low back pain M54.50); each condition carries a
  routine medication (RxNorm ingredient codes) with p = 0.65.

## 3. Streams and linkage

1. **`wearable_daily.csv`** — `patient_id, date, sleep_duration_hours,
   sleep_onset_hour, resting_hr, hrv_rmssd, steps`. One row per patient-day.
2. **`recovery_diary.csv`** — `patient_id, date, recovery_rating` (1–5 integers).
   This is the **prediction target**; ~5% of entries are missing (patient forgot to
   log). Entries are patient-reported in the fiction; the value is produced by the
   label equation below.
3. **`fhir/tb-patient-XXX.json`** — one valid FHIR R4 `collection` Bundle per patient:
   `Patient` (id, gender, birthDate), 4 `Observation` resources (LOINC 39156-5 BMI,
   8480-6 systolic, 8462-4 diastolic, 8867-4 resting HR; UCUM units),
   0–3 `Condition` (ICD-10-CM, onset 2–10 years back), 0–2 `MedicationRequest`
   (RxNorm ingredient codes). All three streams are linked by the synthetic
   `patient_id` and, for the daily streams, by exact date.

## 4. Daily dynamics (wearable stream)

- **Sleep duration:** AR(1) around personal need (coefficient 0.45) + weekend bonus
  (+0.35 h Fri/Sat) + short-sleep events (p = 0.12, −1.9 h) + long events
  (p = 0.06, +1.2 h) + N(0, 0.55).
- **Sleep onset:** 22.6 + 0.9·chronotype + 1.3 (short-sleep nights) + N(0, 0.7),
  clipped [20, 27.5].
- **Sleep-debt effects:** tonight's shortfall below personal need raises resting HR
  (+2.2 bpm per hour of debt) and lowers HRV (−2.4 ms per hour).
- **Illness windows:** 35% of patients get one 2–4 day episode: RHR +U(6, 11) bpm,
  HRV −U(12, 18) ms, steps −50%.
- **Steps:** personal base × weekday profile [Mon..Sun] = [1.05, 1.00, 0.98, 1.00,
  1.04, 1.22, 1.12] × (1 − 0.5·illness) × lognormal(0, 0.22).
- **Missingness:** per-cell sensor dropout p = 0.06; a consecutive 3–5-day "device
  not worn" gap for 18% of patients. Missing values are never interpolated by the
  generator.

## 5. Label-generation process (the assumption, in full)

The recovery rating reported on morning/day `t` is

```
latent_t = base_recovery
         + 0.45 · (sleep_t − sleep_need)                 # last night's sleep
         + 0.30 · (hrv_t − hrv_base) / 8                 # today's HRV
         − 0.25 · (rhr_t − rhr_base) / 4                 # today's resting HR
         + 0.20 · (steps_t − steps_base) / 4000          # today's activity
         − 0.10 · (onset_t − habitual_onset)             # late-night timing
         − Σ condition.recovery_penalty                  # 0.06–0.18 per condition
         − 0.16 · min(n_conditions, 3)                   # burden offset
         − 0.05 · n_medications                          # severity proxy
         − 1.10 · illness_flag_t
         + 0.22 · (latent_{t−1} − 3)                     # day-to-day persistence
         + N(0, 0.45)
recovery_rating_t = clip(round(latent_t), 1, 5)
```

**Reading this honestly:**
- Every coefficient above is an **invented simulator assumption**, not a fitted or
  literature-derived effect. It exists so the prediction task is learnable and the
  ablation (§8 of the brief) is testable.
- The clinical terms (condition penalty, burden, medication proxy) are what couple
  stream B to the label. Without them, the clinical features would provably add
  nothing — the evaluation would then honestly show exactly that.
- The medication term is a stand-in for *overall disease severity*, **not** a
  pharmacological claim.
- Because the AR term uses the latent (unlogged) value, a missed diary entry is a
  *reporting gap*, not a change in the patient's state — matching the fiction that
  the patient forgot to log.

## 6. Known limitations of this dataset

- Daily aggregates only (no intra-night sleep stages, no minute-level HR).
- Static clinical snapshot (conditions/meds/observations have one effective date) —
  no longitudinal clinical events in v1.
- 24 patients is small; ablation deltas of ≲0.05 MAE should be treated as noise.
- The condition→recovery coupling and every dynamic above would need re-derivation
  before any use with real data.
- Synthea-style demographics (races, immunizations, encounters, payers) are out of
  scope in v1; bundles contain only the resources the pipeline consumes.
