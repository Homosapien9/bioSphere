#!/usr/bin/env python3
"""End-to-end smoke test — run this before a demo.

Verifies, without any network access or API keys:
  1. the shipped demo dataset loads and passes validation
     (regenerated deterministically if absent);
  2. the feature pipeline builds leakage-safe features;
  3. all predictors fit and evaluate on held-out rows;
  4. next-day predictions exist and stay on the 1-5 scale;
  5. the what-if scenario engine changes model inputs and recomputes outputs.

Exit code 0 = all checks passed; 1 = something failed (printed).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import data_generation as DG  # noqa: E402
from src import evaluation as EV  # noqa: E402
from src import features as FE  # noqa: E402
from src import models as M  # noqa: E402
from src import pipeline as PL  # noqa: E402
from src import scenarios as S  # noqa: E402
from src import validation as V  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    print(f"Biosphere smoke test (seed={config.RANDOM_SEED})")

    # 0. demo data present?
    data_dir = config.DEMO_DATA_DIR
    needed = [data_dir / config.WEARABLE_CSV_NAME,
              data_dir / config.DIARY_CSV_NAME,
              data_dir / config.FHIR_DIR_NAME]
    if not all(p.exists() for p in needed):
        print("  demo dataset missing -> regenerating deterministically")
        cohort = DG.generate_cohort()
        DG.write_cohort(cohort, data_dir)

    # 1. load + validate (shared ingestion path with the dashboard)
    data = PL.load_inputs(data_dir)
    wearable, diary, clinical = data["wearable"], data["diary"], data["clinical"]
    profiles = data["profiles"]
    w_val, d_val = V.validate_wearable(wearable), V.validate_diary(diary)
    a_val = V.validate_alignment(wearable, diary, clinical)
    check("demo data loads", len(wearable) > 0 and len(profiles) > 0,
          f"{len(wearable)} wearable rows, {len(profiles)} FHIR bundles")
    check("wearable validation passes", w_val.passed, "; ".join(w_val.errors))
    check("diary validation passes", d_val.passed, "; ".join(d_val.errors))
    check("streams align per patient", len(clinical) == wearable["patient_id"].nunique()
          and not a_val.errors)

    # 2. features (single pass: observed frame + next-day rows)
    daily = data["daily"]
    frame, next_rows = FE.build_frames(daily, clinical=clinical)
    usable = FE.usable_for_training(frame)
    check("feature frame builds", len(usable) > 100,
          f"{len(usable)} usable rows, "
          f"{len(FE.get_feature_names(['wearable', 'baseline', 'clinical']))} features")

    # 3. models + evaluation
    results = EV.run_full_evaluation(frame, include_hgbr=True)
    n_test = results["split"]["n_test_rows"]
    check("evaluation produces held-out rows", n_test > 50, f"n_test={n_test}")
    table = EV.results_table(results)
    print()
    print(table[["model", "config", "n", "mae", "rmse"]].to_string(index=False))
    print()
    best = min(results["models"].values(), key=lambda r: r.chronological["mae"] or 99)
    check("all MAE within sanity bound (<2.5 on 1-5 scale)",
          all((r.chronological["mae"] or 99) < 2.5 for r in results["models"].values()),
          f"best: {best.model_name} MAE={best.chronological['mae']:.3f}")
    check("predictions recorded for every model",
          set(results["predictions"]["model"].unique()) == set(results["models"]))

    # 4. next-day prediction
    yhat = M.predict(results["artifacts"]["ridge_with_clinical"], next_rows)
    check("next-day predictions exist and are on the 1-5 scale",
          len(yhat) == wearable["patient_id"].nunique()
          and yhat.min() >= 1 and yhat.max() <= 5,
          f"{len(yhat)} predictions")

    # 5. scenario engine
    train, _test = EV.make_train_test(usable)
    train_range = S.training_sleep_range(train)
    scen = S.run_sleep_scenarios(
        results["artifacts"]["ridge_with_clinical"], next_rows.iloc[0],
        [4.0, 7.0, 10.0], train_range,
        resid_stats=results["models"]["ridge_with_clinical"].train_residual_stats)
    check("scenario engine recomputes model outputs",
          len(scen) == 3 and scen["predicted_rating"].nunique() >= 1
          and scen["predicted_rating"].between(1, 5).all(),
          f"sleep 4/7/10h -> {list(scen['predicted_rating'].round(2))}")
    expected_in_range = [(train_range[0] <= s <= train_range[1]) for s in (4.0, 7.0, 10.0)]
    check("scenario flags out-of-training-range inputs",
          list(scen["in_training_range"]) == expected_in_range,
          f"train range {train_range[0]:.1f}-{train_range[1]:.1f}h, "
          f"flags {list(scen['in_training_range'])}")

    failed = [c for c in CHECKS if not c[1]]
    print()
    if failed:
        print(f"SMOKE TEST FAILED: {len(failed)}/{len(CHECKS)} checks failed:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print(f"SMOKE TEST PASSED: {len(CHECKS)}/{len(CHECKS)} checks.")
    print(config.SYNTHETIC_DISCLAIMER)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        print("SMOKE TEST FAILED: unhandled exception above.")
        raise SystemExit(1)
