# Biosphere — 90-Second Demo Script

> Speaker note: every number shown on screen is computed live by the pipeline from
> the shipped synthetic dataset. If asked "is this real data?" the answer is the
> banner at the top of the app: **100% synthetic, research prototype, not a medical
> device.** That honesty is part of the pitch, not a caveat to hide.

**[0:00–0:12] Hook.**
"Every digital-twin demo you've seen today shows a beautiful dashboard. The hard
question is never the UI — it's: *does the personalization actually beat a dumb
baseline, and can you prove it without leaking future data?* Biosphere answers that
with a tested pipeline, not a promise."

**[0:12–0:30] Two streams, one patient.**
Open **Overview**. Point at the FHIR clinical profile on the left — "a real FHIR R4
bundle, parsed with Pydantic validation: age, conditions in ICD-10, medications in
RxNorm" — and wearable completeness on the right: "90 days of sleep, HRV, resting
heart rate, steps, with missing days left *visible* — we never invent measurements."
Emphasize: same synthetic patient id on both sides; all three views ship in the repo.

**[0:30–0:48] The prediction, with receipts.**
Open **Prediction**. "For tomorrow, the persistence baseline says 3.0 — tomorrow
equals today. Our personalized Ridge model says 2.57 — it used last night's sleep
deviation from *this patient's own* baseline, HRV, resting HR, and the clinical
context. Every feature comes from before the target day — we enforce that with a
perturbation test that literally corrupts the future and checks nothing changes."

**[0:48–1:08] The honest evaluation.**
Open **Model evaluation**. "Chronological split — last 25% of days held out, never
shuffled — plus a held-out-patient experiment. Result: personalized Ridge 0.644 MAE
versus persistence 0.697. And read this row: wearable-*only* is *worse* than
persistence, and the clinical stream added nothing measurable at this cohort size.
We show that instead of hiding it — honest negative results are the benchmark."

**[1:08–1:26] The bounded what-if.**
Open **What-if explorer**. "One scenario, deliberately bounded: move tonight's sleep
from 6.3 to 4 hours — the fitted model recomputes 2.57 → 2.29. Outside the training
range, it warns you. It re-runs the actual model — no animations, no hard-coded
scores — and it's labeled a hypothetical association, not a causal claim."

**[1:26–1:35] Trust & engineering close.**
Back to any tab. "80 unit tests, an 11-check smoke test, zero API keys, one command
to run. Everything is seeded and reproducible, the dataset card documents the exact
label equation, and the audit of the three repos that inspired the design is in
/docs. Biosphere — a digital twin you can actually *test*."

---

## Contingency notes

- If the app is slow on first load: the model-fitting spinner appears once (~5–8 s),
  then everything is cached.
- If asked for real data: dataset card §1 documents the verified availability of
  Synthea / WESAD / BIDSleep and why none pairs wearables + EHR + daily self-ratings;
  the wearable loader accepts any CSV with the same schema.
- If asked for live vendor data: deliberately out of scope in v1 (audit §5);
  the loader is the designed plug-in point.
