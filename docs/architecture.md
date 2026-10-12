# Biosphere — Architecture

## 1. One-paragraph summary

Biosphere is a single-process, offline, Python 3.11+ research prototype. A
deterministic simulator emits three linked views of 24 synthetic patients (daily
wearable CSV, daily recovery diary CSV, one FHIR R4 bundle per patient). Loaders
parse and validate them, a feature builder produces strictly past-only features per
patient-day, an evaluation engine compares two transparent baselines against
regularized regression under a per-patient chronological split plus a held-out-patient
experiment, and a Streamlit dashboard renders predictions, evaluation, a bounded
sleep what-if, and the full evidence trail. There are no services, no network calls,
no API keys, and no LLM anywhere.

## 2. Data flow

```
scripts/generate_demo_data.py
        │  (deterministic, seed=20260214)
        ▼
data/demo/wearable_daily.csv      data/demo/recovery_diary.csv     data/demo/fhir/*.json
        │                                  │                             │
        └──────────── PL.load_inputs ──────┴─────────────────────────────┘
                     (src/pipeline.py — the ONE ingestion path shared by the
                      dashboard, evaluation script, and smoke test)
        │                                  │                             │
        ▼                                  ▼                             ▼
src/wearable_loader.py          src/wearable_loader.py          src/fhir_loader.py
(coerce + transparent masking)  (diary ratings; invalid →       (Pydantic R4 models,
        │                        reported & dropped)             strict bundle checks)
        │                                  │                             │
        └────────────┬─────────────────────┘                             │
                     ▼                                                   │
              src/validation.py   (errors → stop; warnings → UI)          │
                     │                                                   │
                     ▼                                                   │
        src/features.py  merge_streams → build_frames ───────────────────┘
                (every feature at row t uses data < t — shift(1) before
                 rolling/expanding; missing indicators; *_hist_n metadata;
                 one pass yields observed rows AND next-day rows)
                     │
      ┌──────────────┼───────────────────────────┐
      ▼              ▼                           ▼
src/personalization.py   src/models.py      src/evaluation.py
(robust personal          persistence /      chronological split per patient,
 baselines for the         personal_mean /   held-out-patient mask, metrics,
 dashboard & evidence)     Ridge / HGBR      missing-subset metrics
                           (joblib save)          │
                                                  ▼
                                     results/metrics.json,
                                     results/predictions.csv,
                                     docs/evaluation_report.md
                                                  │
                                                  ▼
                                     app.py (Streamlit) — sections A–F
                                     src/scenarios.py — bounded sleep what-if
```

## 3. Leakage policy (the core invariant)

- **Row contract:** the row dated `t` predicts the rating *on* day `t`; every feature
  is built from data strictly *before* `t`. Wearable and diary values are shifted by
  one day *before* any rolling or expanding statistic is applied.
- **Fit-time isolation:** the imputer and scaler live inside a scikit-learn
  `Pipeline` fitted on training rows only; the baselines' fallback (cohort mean) is
  fitted on training targets only.
- **Split design:** per patient, the *last* 25% of usable days are held out — no
  random shuffling of overlapping windows. Every predictor (baselines included) is
  scored on the identical test rows. Additionally, every 5th patient (sorted id) is
  removed from training entirely to measure cold-start behavior.
- **Next-day rows** are produced by appending a valueless placeholder day and reusing
  the same shift-based builder, so live predictions cannot see anything newer than
  the last observed day.
- **Enforced by tests:** `tests/test_features.py::test_no_future_leakage_perturbation_test`
  perturbs the future and asserts past features are bit-identical.

## 4. Model rationale

- **Persistence** (last rating) and **personal expanding mean** are the two baselines
  the brief mandates; they are stateless given the feature row, cheap, and hard to
  beat on a 1–5 scale with day-level noise.
- **Ridge** (median-impute → standardize → L2) is the primary model: interpretable,
  deterministic, appropriate for ~1,900 rows / ~47 features, and natively supported
  by per-coefficient inspection if needed.
- **HistGradientBoostingRegressor** is run as a *comparison only* and adopted solely
  if it improves held-out MAE; on the shipped data it does not (0.689 vs 0.650), so
  it is reported as not adopted.
- **Uncertainty** is an explicitly labeled proxy: ±1.96·σ and empirical 2.5/97.5
  residual percentiles from training residuals, clipped to the 1–5 scale ("error
  visualization, not a calibrated interval" — wording adopted from BioTwin's model
  card discipline).

## 5. What was adopted / simplified / excluded from the three audited repos

Full audit: `docs/repository_audit.md`.

| Decision | Item | Origin |
|---|---|---|
| **Adopted (idea)** | Synthetic-data provenance labels on every surface | BioTwin |
| **Adopted (idea)** | Refuse-to-forecast; report imputed inputs; missing stays missing | BioTwin |
| **Adopted (idea)** | Robust personal baselines with sufficiency flags (median/MAD) | BioTwin (simplified) |
| **Adopted (idea)** | Chronological/prequential evaluation discipline | BioTwin |
| **Adopted (idea)** | Pydantic-typed FHIR R4 resource models | PatientPulse (MIT; re-implemented for local bundles) |
| **Adopted (idea)** | Ship-with-the-repo static demo data + fallback path | PatientPulse (maria_static.json pattern) |
| **Adopted (idea)** | Deterministic engine for all numbers; no LLM math (we go further: no LLM at all) | PatientPulse |
| **Adopted (idea)** | Slider-driven what-if over a fitted model, with bounds warnings | VitalCast UX (implementation fixed) |
| **Simplified** | One Streamlit app instead of FastAPI + React + nginx | — |
| **Simplified** | Local FHIR bundles instead of HAPI FHIR server + Postgres + Redis | — |
| **Simplified** | scikit-learn Ridge instead of MATLAB-exported coefficients; daily aggregates instead of minute-level streams | — |
| **Excluded** | LLM agents, voice, 3D avatar, OAuth vendor integrations, Docker orchestration, medication-response simulation, disease-risk scores | all three |
| **Reused as code** | Nothing — zero code copied from any repo (BioTwin & VitalCast have no license; PatientPulse is MIT but patterns were re-implemented) | — |

## 6. Repository layout

```
biosphere/
├── app.py                    # Streamlit dashboard (sections A–F)
├── config.py                 # every tunable constant in one place
├── requirements.txt
├── data/demo/                # shipped synthetic dataset (CSVs + 24 FHIR bundles)
├── src/
│   ├── pipeline.py           # single ingestion path shared by app + scripts
│   ├── data_generation.py    # deterministic cohort + FHIR bundle generator
│   ├── wearable_loader.py    # CSV load, coercion, transparent masking
│   ├── fhir_loader.py        # Pydantic R4 parsing → clinical feature table
│   ├── validation.py         # error/warning gates
│   ├── features.py           # past-only feature engineering (leakage-safe)
│   ├── personalization.py    # robust personal baselines (dashboard layer)
│   ├── models.py             # baselines + Ridge (+ HGBR), joblib persistence
│   ├── evaluation.py         # splits, metrics, ablations, held-out patients
│   ├── uncertainty.py        # labeled residual-based proxy
│   └── scenarios.py          # bounded sleep what-if engine
├── tests/                    # 80 tests across 11 files
├── scripts/
│   ├── generate_demo_data.py
│   ├── run_evaluation.py     # writes results/ + docs/evaluation_report.md
│   └── smoke_test.py         # pre-demo end-to-end check
└── docs/                     # this file, dataset_card, repository_audit,
                              # evaluation_report (generated), demo_script
```

## 7. Deliberate v1 simplifications

- Clinical context is a static per-patient snapshot (conditions/meds/observations do
  not evolve over the 90 days).
- `condition_burden` uses fixed severity weights shared with the simulator's library;
  it is a documented proxy, not a clinical index.
- The evaluation universe is restricted to days with ≥1 prior rating so the
  persistence baseline is defined; all models share these identical rows.
- Uncertainty ignores parameter uncertainty and distribution shift; it is a proxy by
  design and labeled as such.
