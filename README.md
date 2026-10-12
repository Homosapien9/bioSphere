# Biosphere 🌿

**Your physiology, understood over time.**

*Biosphere is an experimental personalized physiological digital-twin prototype:
it forecasts next-day recovery ratings from wearable time series + structured
FHIR R4 clinical context.*

Biosphere joins two data streams for the *same* synthetic patient — daily wearable
records and a valid FHIR R4 clinical bundle — and honestly measures whether
personalizing a forecast on top of simple baselines actually works. It is a
**research and education prototype, not a medical device**: all data are synthetic,
all outputs are model outputs, and nothing here is medical advice, diagnosis, or a
causal estimate.

> **Status: working end-to-end and tested.** 80/80 unit tests pass, an 11-check
> smoke test passes, the dashboard boots and renders verified in a headless browser.
> Every number in the app and in `docs/evaluation_report.md` was computed by running
> the actual pipeline on the shipped dataset. Peak dashboard memory measured at
> ~0.3 GB — comfortably inside a 1.5 GB deployment budget.

---

## Quickstart (one command, no keys, no network)

```bash
pip install -r requirements.txt
streamlit run app.py
```

That's it — the synthetic demo dataset (24 patients × 90 days, wearable CSV +
recovery diary CSV + 24 FHIR R4 bundles) ships inside the repo, and the app makes
zero network calls.

Regenerate data / evaluate / verify:

```bash
python scripts/generate_demo_data.py      # regenerate the demo dataset (same seed → identical)
python scripts/run_evaluation.py          # train + evaluate everything; writes docs/evaluation_report.md
python scripts/smoke_test.py              # 11-check end-to-end pre-demo verification
python -m pytest tests/ -q                # unit tests
```

## What the app shows

| Tab | Contents |
|---|---|
| **Overview** | FHIR-derived clinical profile (age, sex, BMI, BP, conditions, medications) beside wearable completeness for the same patient; data provenance + validation report |
| **Physiological history** | Sleep / HRV / resting-HR / steps time series with personal-median baselines, recent-window highlights, and missing-data gaps left visible |
| **Prediction** | Next-day recovery-rating estimate from the fitted Ridge model **side-by-side** with persistence and personal-mean baselines; uncertainty proxy; data-sufficiency; exact model inputs |
| **Model evaluation** | Held-out MAE/RMSE for all 6 predictors, predicted-vs-actual scatter, per-patient MAE, the mandated ablation (wearable-only → +history → +clinical), missing-data subsets, held-out-patient cold start, HGBR adoption verdict |
| **What-if explorer** | Bounded sleep scenario: the fitted model is re-run with tonight's sleep replaced; out-of-range warnings; labeled as hypothetical, not causal |
| **Evidence & limitations** | The 14-day feature window, the FHIR resources consumed, every missing input, and the full limitation list |

## The headline (honest) numbers

From `docs/evaluation_report.md` — chronological held-out test rows (n=393 days,
19 train patients), synthetic data, MAE on the 1–5 rating scale:

| model | MAE ↓ | RMSE | held-out-patient MAE |
|---|---|---|---|
| persistence baseline | 0.697 | 0.960 | 0.659 |
| personal rolling mean | 0.751 | 0.926 | 0.706 |
| Ridge wearable-only | 0.806 | 0.998 | 0.730 |
| **Ridge + personal history** | **0.644** | **0.816** | **0.634** |
| Ridge + history + clinical | 0.650 | 0.820 | 0.845 |
| HistGradientBoosting (comparison) | 0.689 | 0.876 | 0.681 |

What these numbers honestly say: the personalized regression **does beat the
persistence baseline** (0.644 vs 0.697, ~7.6% relative MAE reduction); a wearable-only
model (no rating history) is *worse* than persistence; adding FHIR clinical context
showed **no measurable improvement at this cohort size** — a negative result we
report rather than manufacture (the simulator does plant a clinical association; the
model's personal-history features absorb it). Full discussion in the evaluation report.

## Verification evidence

- `python -m pytest tests/ -q` → **80 passed** (data generation, loaders, FHIR,
  validation, leakage prevention, models, evaluation, scenarios, uncertainty,
  end-to-end app data path)
- `python scripts/smoke_test.py` → **11/11 checks passed**
- `streamlit run app.py` → HTTP 200 + health check OK, rendered in headless Chromium
  across all six tabs, patient switching and the what-if slider exercised with no
  exceptions (screenshots in `docs/`)
- Memory: peak resident memory ~0.30–0.35 GB during a full tab-walk + interaction
  session — far inside the 1.5 GB budget
- Determinism: `generate_cohort(seed)` produces byte-identical outputs (tested)
- BioTwin's own test suite (audited dependency) ran in an isolated venv: 153 passed /
  3 skipped / 3 failed (failures need a configured PostgreSQL) — see
  `docs/repository_audit.md`

## Repository map

```
biosphere/
├── app.py                 # Streamlit dashboard (six tabs)
├── config.py              # all constants: seed, splits, bounds, feature sets
├── src/                   # pipeline modules (see docs/architecture.md)
├── tests/                 # 11 test files, 80 tests
├── scripts/               # generate_demo_data / run_evaluation / smoke_test
├── data/demo/             # shipped synthetic dataset (CSVs + FHIR bundles)
├── results/               # metrics.json, predictions.csv, ridge_with_clinical.joblib
└── docs/                  # architecture, dataset_card, repository_audit,
                           # evaluation_report (generated), demo_script, screenshots
```

## Design boundaries (deliberate)

Included: deterministic synthetic generator · FHIR R4 ingestion with Pydantic
validation · leakage-safe features (enforced by a perturbation test) · transparent
baselines vs Ridge vs (optional) HGBR · chronological + held-out-patient evaluation ·
labeled uncertainty proxy · bounded sleep what-if · provenance on every surface.

Excluded from this release: LLMs/agents, live wearable OAuth, production EHR
connections, Docker orchestration, medication/treatment simulation, disease
diagnosis, 3D avatars, cloud anything. See `docs/architecture.md` §5 for the
adopt/simplify/exclude decision table derived from the three audited repositories.

## Attribution & licenses

Ideas were adopted (never code) from three audited hackathon projects — BioTwin,
PatientPulse, VitalCast AI — and are credited in
[`LICENSES_AND_ATTRIBUTION.md`](LICENSES_AND_ATTRIBUTION.md), which also records the
license status that governed what could be reused. Biosphere's own code is MIT.
FHIR R4 is used purely as a data format (HL7® FHIR® is an HL7 specification; no HL7
artifacts are redistributed).

## Known limitations

- Everything is synthetic; the label equation is an invented simulator assumption
  (`docs/dataset_card.md` §5). No result here transfers to real patients.
- Clinical context is a static snapshot; 24 patients is small (deltas ≲0.05 MAE are
  noise); the clinical stream's value at cold start is unproven at this scale.
- Uncertainty bands are residual-based proxies, not calibrated intervals.
- The what-if explorer changes one input and holds the rest fixed — an association
  view, explicitly not physiology simulation and not a demonstrated causal effect.
