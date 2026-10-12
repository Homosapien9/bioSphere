# Phase 1 — Repository Audit (BioTwin, PatientPulse, VitalCast AI)

Audit date: 2026-10-12 · Method: all three repositories were cloned (`git clone --depth 1`),
their READMEs, dependency manifests, licenses, and source files were read directly, and — where
feasible — their test suites were actually executed. Findings below are limited to what was
observed in code and verified runs; nothing is inferred from README claims alone.

---

## 1. BioTwin — https://github.com/HackRice2026/BioTwin (HackRice 2026)

**Stack (verified):** Python 3.12, FastAPI + SQLAlchemy (+ PostgreSQL in production), NumPy/SciPy
only (no pandas/scikit-learn), offline model fitting in MATLAB with coefficients exported to JSON,
two frontends (React/Vite PWA and a Next.js Garmin dashboard), a Garmin Monkey-C watch app,
Docker/Caddy ops, Supabase schema migration. 156 Python tests collected.

**What is actually implemented (read in code):**
- `ingestion/adapters/` — a uniform ingestion pipeline with adapters for Garmin, Fitbit, `.FIT`
  file import, recorded replay, a **deterministic explicitly-synthetic adapter** (`synthetic.py`,
  seeded `np.random.default_rng(42)`), watch BLE, and InfluxDB sync. Every frame carries a
  provenance label.
- `modeling/vitals_history.py` / `training_window.py` — weighted robust personal baselines
  (28-day trailing window, 7-day half-life exponential weighting, weighted median/MAD, shrinkage
  of limited observations toward explicit engineering priors).
- `modeling/recovery.py` — heart-rate-recovery exponential time-constant fitting
  (`scipy.optimize.curve_fit`) with bounded parameters, saturation detection ("a constant resting
  against its bound is an unidentified fit, not a slow one"), and **leave-one-session-out
  validation**; ≥3 accepted sessions required before any personal tau is issued.
- `modeling/forecast.py` — a Body-Battery ridge regression whose standardized coefficients were
  fitted offline in MATLAB and exported as plain JSON (`matlab/params_forecast.json`); the runtime
  has zero MATLAB dependency. Refuses to forecast when the current reading is stale ("a stale
  level would produce a confidently wrong number. Rather than extrapolate from one, the forecast
  is refused and says why"), reports which inputs were **imputed**, and — for longer horizons —
  switches predictors per horizon by validation error, attaching each point's method and error.
- **Prequential evaluation**: predictions are stored *before* their observations; the original
  forecast is immutable and scored later only over post-issuance samples with matching provenance
  (`recovery.py::score_prediction`).
- `replay` path that re-runs the fitted model at the last real sample and excludes later frames,
  explicitly "to prevent future information from leaking into that historical run."

**Advertised vs. real:** README features and code agree. Credential-gated features (Garmin cloud,
Fitbit/Google Health, Google Calendar, Gemini/Vertex, ElevenLabs) degrade to explicit
"setup required" states rather than fake data.

**Tests — actually run in an isolated venv (this audit):** `pytest -q` →
**153 passed, 3 skipped, 3 failed**. The 3 failures are `sqlalchemy.exc.ArgumentError: Could not
parse SQLAlchemy URL` — they require a configured PostgreSQL database; BioTwin's own
`docs/VERIFICATION.md` notes PostgreSQL-only cases are skipped locally without CI's database.
No logic failures observed.

**Setup blockers for a beginner:** `uv` + Node 22 toolchain, `.env` with provider credentials for
anything live, PostgreSQL in production deployment, webhook configuration for cloud ingestion.

**License:** **no root LICENSE file** (only `docs/LICENSES.md`, which covers assets and vendored
npm/Python dependencies). Absent a license, default is all-rights-reserved → **we reuse ideas
only, zero code**.

---

## 2. PatientPulse — https://github.com/pprenapurkar/Patientpulse (ScarletHacks 2026, 2nd place)

**Stack (verified):** FastAPI (Python 3.11) + `anthropic` SDK for a Claude multi-agent layer,
HAPI FHIR R4 server, PostgreSQL, Redis, React 18 + Vite + TypeScript frontend, Docker Compose
with nginx. `backend/requirements.txt` pins fastapi 0.111, pydantic 2.7, anthropic 0.29, redis.

**What is actually implemented (read in code):**
- `backend/app/services/fhir_service.py` — a typed async HAPI FHIR client
  (Patient/Condition/MedicationStatement/Observation/AllergyIntolerance/Encounter searches with
  explicit `FHIRServerUnavailableError` / `PatientNotFoundError`).
- `backend/app/schemas/fhir_types.py` — **Pydantic v2 models for FHIR R4 resources** with
  camelCase aliases and `Literal["Condition"]`-style resourceType guards, plus field validators
  (e.g., `subject.reference` must start with `Patient/`). This is the strongest single file in
  the repo for our purposes.
- `fixtures/maria_static.json` + `FHIRFallbackBanner.tsx` — a **static FHIR fixture and UI
  fallback banner** so the demo still works when the FHIR server is down. A good reliability
  pattern for demos.
- `backend/app/services/wearable_ingester.py` — generates 72 h of simulated wearable data and
  writes it as real FHIR Observation resources (LOINC codes, UCUM units). Note: uses unseeded
  `random`, so output is not reproducible — an anti-pattern we must avoid.
- `ai/engines/projection_engine.py` — deterministic 12-week HbA1c/weight projection from
  literature-coefficient linear rules, headlined "**NO LLM math**".
- `ai/agents/scenario_agent.py` — explicit ordering: deterministic RxNorm check → deterministic
  projection → LLM produces narrative *only*. "LLM is NEVER used for numeric computation."

**Advertised vs. real:** README's "ML-based triage" is, in code, rule-based thresholds plus LLM
classification; "evidence-based HbA1c projections" are fixed linear coefficients (hard-coded,
e.g. `current_weight_kg = 78.0` with a comment pinning it to the demo patient). Acceptable for a
48-hour hack, but the README language overstates it.

**Tests:** the `pyproject.toml` ruff config references `tests/**` per-file ignores, but **no test
files exist anywhere in the repository** (verified by listing and search). Nothing to run.

**Setup blockers:** `ANTHROPIC_API_KEY` is mandatory for most flows; Docker Desktop; HAPI FHIR
container needs ~2 minutes to initialize; RxNorm calls need network at runtime.

**License:** **MIT** (verified in `LICENSE`) — code *may* be adapted with attribution; we still
chose to re-implement independently and cite the patterns.

---

## 3. VitalCast AI — https://github.com/ethnmz/VitalCast-AI

**Stack (verified):** pure client-side React 18 + Vite (JavaScript), Chart.js, jsPDF, Web Worker
for parsing 800 MB Apple Health exports with regex in-browser; no backend, no Python.

**What is actually implemented (read in code):**
- `src/lib/parseAppleHealth.js` / `parseGarmin.js` / `parserWorker.js` — genuinely client-side
  parsing of real wearable exports; privacy-first (data never leaves the browser).
- `src/lib/models.js` — deterministic scoring: HRV normalization to 0–100, a *simplified* risk
  score labeled "Framingham", a PSQI approximation from average sleep hours, a stress index
  (100 − HRV score), a 30-day linear projection with a fan-out uncertainty band.
- `src/lib/claudeApi.js` — calls the Anthropic API **directly from the browser** with
  `VITE_ANTHROPIC_API_KEY` (including the `anthropic-dangerous-direct-browser-access` header) —
  the key ships to every visitor; without a key, narrative features fail.
- A 3-act guided narrative UI (`Act0Explained` → `Act1Findings` → `Act2Simulator` → `Act3Plan`)
  with what-if sliders ("Active Life Gained").

**Credibility red flags (documented honestly, and directly relevant to our design):**
- `historical14Days()` **fabricates a "14-day historical trend" using `Math.random()`** —
  invented history drawn to look measured.
- `computeMetrics()` **silently replaces missing or out-of-range values with hard-coded sample
  defaults** (resting HR 85, HRV 32, a 52-year-old smoker profile) — measurements are invented
  rather than reported missing.
- The "Framingham CVD Risk Score" is an ad-hoc point score, not the validated equation.
- `modelConfidence()` conflates data completeness with model validity and displays it as a
  confidence percentage.

**Tests:** none — no test files and no `test` script in `package.json`.

**License:** **none found** → all-rights-reserved; ideas only, zero code reuse.

---

## 4. Comparison across the eight required categories

| Category | BioTwin | PatientPulse | VitalCast AI |
|---|---|---|---|
| **1. Personalized wearable baselines & time-series forecasting** | Strongest: weighted robust baselines (median/MAD, shrinkage), recovery tau with LOSO validation, ridge forecast from exported coefficients, per-horizon predictor selection by validation error | None — no fitted models; rule thresholds only | Simple static score transforms; linear trend projection; no personalization beyond averages |
| **2. Clinical/EHR integration** | None — wearables only | **Strongest**: real FHIR R4 client + Pydantic R4 resource models + HAPI server + static fixture fallback | None |
| **3. What-if scenario modeling** | Day simulation on fitted tau; "fixed illustrative assumptions, not measured intervention effects" (labeled) | Scenario agent = deterministic projection + LLM narrative; medication scenarios (out of our scope) | **Strongest UX**: interactive sliders over a deterministic model; but no out-of-range guards |
| **4. Explainability & uncertainty** | **Strongest**: per-point validation error bands labeled "error visualization, not calibrated interval"; imputed inputs disclosed; refusal with reasons | Deterministic/LLM separation is its explainability story; no numeric uncertainty | A "confidence %" that conflates completeness with validity; CI width from that same heuristic |
| **5. ML evaluation & testing** | 156 tests; prequential scoring; LOSO validation; we ran: **153 pass / 3 skip / 3 fail (DB-config)** | **Zero tests present** (config references a tests dir that does not exist) | **Zero tests** |
| **6. UI & demonstration quality** | Polished PWA, 3D avatar, voice; heavy | Two polished dashboards (clinician/patient), mobile-first | Polished narrative "acts"; what-if UX is the standout |
| **7. Installation complexity & reliability** | High: uv + Node 22 + optional many API keys + Postgres for prod | High: Docker Compose, HAPI (~2 min), Redis, Postgres, **mandatory Anthropic key** | Low: `npm install && npm run dev`; but needs an Anthropic key for narrative |
| **8. Reusable / adaptable features** | Provenance discipline; refuse-to-forecast; prequential evaluation; model-card language; baseline shrinkage; synthetic seeded adapter | Pydantic FHIR resource models; static-FHIR demo fallback; deterministic-engine-before-LLM boundary; FHIR Observation authoring pattern | Client-side parsing architecture; slider→model what-if UX; completeness→sufficiency messaging (idea only) |

---

## 5. Architecture decision for Biosphere

**Adopt (as design ideas, re-implemented from scratch):**
1. **Provenance & synthetic labeling everywhere** (BioTwin): every surface that shows data or a
   prediction is labeled with its source; synthetic data is never dressed as real.
2. **Refuse-to-forecast + disclosure** (BioTwin): if required inputs are missing/stale, show a
   reason, never a fabricated number; every imputed input is reported.
3. **Robust personal baselines with sufficiency flags** (BioTwin, simplified): median/MAD-based
   personal reference values, an explicit "insufficient history" state, no disease claims.
4. **Chronological/prequential evaluation** (BioTwin): time-ordered splits; predictions compared
   against held-out observations only; model metadata saved with the metrics.
5. **FHIR R4 ingestion with Pydantic validation** (PatientPulse): typed resource models with
   `resourceType` literal guards and reference validation — but reading **local FHIR bundle
   JSON files** instead of standing up a HAPI server.
6. **Static-bundle demo fallback** (PatientPulse): the demo dataset ships in the repo, so no
   service must be reachable at demo time.
7. **Deterministic-numbers / no-LLM boundary** (PatientPulse): we go further — no LLM anywhere
   in v1; all numbers come from pandas/scikit-learn code that tests can verify.
8. **Slider-driven what-if over a fitted model** (VitalCast UX), fixed implementation: recompute
   model features, warn outside the training range, label outputs as hypothetical scenarios.

**Simplify:**
- One Streamlit app instead of FastAPI + React + nginx (single-command local demo).
- Local FHIR JSON bundles instead of HAPI server + PostgreSQL + Redis.
- scikit-learn Ridge instead of MATLAB-exported coefficients; daily aggregates instead of
  minute-level streams; residual-based uncertainty proxy instead of calibrated intervals.

**Exclude (per scope control, §14):** LLM agents and chat, voice, 3D avatars, OAuth/live vendor
integrations, Docker multi-service orchestration, medication-response simulation, disease-risk
scores (Framingham-style), readmission/emergency prediction.

**Reuse-license outcome:** BioTwin and VitalCast have no repository license → ideas only.
PatientPulse is MIT → patterns may be adapted with attribution; we still re-implemented
independently and credit it in `LICENSES_AND_ATTRIBUTION.md`.

**Prediction-target & dataset decision (Phase 2 input):** the verification step confirmed
- Synthea: Apache-2.0 (license verified from the raw LICENSE file) — but it generates clinical
  records only, no wearable streams and no daily self-reported recovery ratings.
- WESAD: confirmed via the UCI page (15 subjects, lab stress-affect study, physiological + motion
  sensors) — no EHR pairing, no daily recovery self-ratings, license deferred to a linked page.
- BIDSleep: the PhysioNet URL from the brief (both `1.0.1` and the bare path) currently returns
  **HTTP 404** — the source could not be verified at all at audit time.

No public dataset pairs the same person's daily wearable history, structured EHR context, and a
daily 1–5 self-reported recovery rating under a key-free, small-download license. Per the brief's
fallback provision, v1 therefore uses a **deterministic, seeded synthetic generator** that emits
wearable CSVs, FHIR R4 bundles, and a recovery diary with a fully documented label equation
(`docs/dataset_card.md`). The wearable loader accepts any CSV matching the documented schema so a
real export can be plugged in later without touching the pipeline.
