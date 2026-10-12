#!/usr/bin/env python3
"""Generate the shipped demo dataset (or any cohort) and validate it.

Usage:
    python scripts/generate_demo_data.py                       # -> data/demo
    python scripts/generate_demo_data.py --patients 40 --days 120 \
        --seed 42 --out data/generated
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import data_generation as DG  # noqa: E402
from src import pipeline as PL  # noqa: E402
from src import validation as V  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patients", type=int, default=config.DEMO_N_PATIENTS)
    ap.add_argument("--days", type=int, default=config.DEMO_N_DAYS)
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out", type=Path, default=config.DEMO_DATA_DIR)
    args = ap.parse_args()

    print(f"Generating {args.patients} patients x {args.days} days "
          f"(seed={args.seed}) -> {args.out}")
    cohort = DG.generate_cohort(n_patients=args.patients, n_days=args.days,
                                start_date=config.DEMO_START_DATE, seed=args.seed)
    paths = DG.write_cohort(cohort, args.out)
    print(f"  wearable csv : {paths['wearable_csv']}")
    print(f"  diary csv    : {paths['diary_csv']}")
    print(f"  fhir bundles : {paths['n_bundles']} files in {paths['fhir_dir']}")

    # Validate through the exact path the app will use at runtime.
    data = PL.load_inputs(args.out)
    wearable, diary = data["wearable"], data["diary"]
    profiles, clinical = data["profiles"], data["clinical"]

    w_rep = V.validate_wearable(wearable)
    d_rep = V.validate_diary(diary)
    a_rep = V.validate_alignment(wearable, diary, clinical)
    for label, rep in [("wearable", w_rep), ("diary", d_rep), ("alignment", a_rep)]:
        status = "PASS" if rep.passed else "FAIL"
        print(f"  validation[{label}] {status}: "
              f"{len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
        for e in rep.errors:
            print(f"    ERROR: {e}")
        for w in rep.warnings[:3]:
            print(f"    warn : {w}")
        if len(rep.warnings) > 3:
            print(f"    ... and {len(rep.warnings) - 3} more warnings "
                  f"(day-gap warnings are expected with simulated device gaps)")

    print(f"  patients: {wearable['patient_id'].nunique()} | "
          f"wearable rows: {len(wearable)} | diary rows: {len(diary)} | "
          f"FHIR profiles: {len(profiles)}")
    if not (w_rep.passed and d_rep.passed):
        print("FATAL: validation reported errors.")
        return 1
    print("Demo dataset generated and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
