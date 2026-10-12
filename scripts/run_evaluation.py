#!/usr/bin/env python3
"""Train every predictor on the demo data and write the evaluation report.

Usage:
    python scripts/run_evaluation.py                 # demo data -> results/
    python scripts/run_evaluation.py --data-dir data/generated

Outputs:
    results/metrics.json         machine-readable metrics
    results/predictions.csv      held-out predictions for every model
    docs/evaluation_report.md    human-readable report (actual numbers only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import evaluation as EV  # noqa: E402
from src import features as FE  # noqa: E402
from src import models as M  # noqa: E402
from src import pipeline as PL  # noqa: E402
from src import validation as V  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=config.DEMO_DATA_DIR)
    ap.add_argument("--skip-hgbr", action="store_true",
                    help="skip the HistGradientBoosting comparison")
    args = ap.parse_args()

    print(f"Loading data from {args.data_dir}")
    data = PL.load_inputs(args.data_dir)
    wearable, diary, clinical = data["wearable"], data["diary"], data["clinical"]
    frame = FE.build_feature_frame(data["daily"], clinical=clinical)
    V.require_passed(V.validate_wearable(wearable), "wearable")
    V.require_passed(V.validate_diary(diary), "diary")
    align_rep = V.validate_alignment(wearable, diary, clinical)
    usable = FE.usable_for_training(frame)
    print(f"  usable rows: {len(usable)} | patients: {usable['patient_id'].nunique()}")
    print(f"  load masking: wearable {data['w_report'].masked_implausible or 'none'}, "
          f"diary {data['d_report'].masked_implausible or 'none'}; "
          f"alignment warnings: {len(align_rep.warnings)}")

    print("Running full evaluation (chronological + held-out patients)...")
    results = EV.run_full_evaluation(frame, include_hgbr=not args.skip_hgbr)
    table = EV.results_table(results)
    print()
    print(table.to_string(index=False))

    # Persist artifacts. Record the data dir relative to the project root so
    # generated artifacts never leak local absolute paths into the release.
    config.RESULTS_DIR.mkdir(exist_ok=True)
    try:
        data_dir_rel = str(args.data_dir.resolve().relative_to(ROOT))
    except ValueError:
        data_dir_rel = str(args.data_dir)
    metrics = {
        "data": {
            "dir": data_dir_rel,
            "synthetic": True,
            "n_usable_rows": int(len(usable)),
            "n_patients": int(usable["patient_id"].nunique()),
            **results["split"],
        },
        "models": {name: {
            "kind": r.kind, "config": r.config,
            "chronological": r.chronological,
            "heldout_patients": r.heldout_patients,
            "missing_subsets": r.missing_subsets,
            "train_residual_stats": r.train_residual_stats,
        } for name, r in results["models"].items()},
        "hgbr_adoption": results["hgbr_adoption"],
    }
    (config.RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if len(results["predictions"]):
        results["predictions"].to_csv(config.RESULTS_DIR / "predictions.csv", index=False)

    # Persist the primary model artifact (refit on ALL usable data) for reuse.
    all_rows = FE.usable_for_training(frame)
    primary = results["models"].get("ridge_with_clinical")
    if primary is not None:
        refit_all = M.fit_sklearn("ridge", primary.artifact["config"], all_rows)
        M.save_artifact(refit_all, config.RESULTS_DIR / "ridge_with_clinical.joblib",
                        extra_meta={"refit_scope": "all usable rows (deployment model)",
                                    "eval_scope": "chronological test split of demo data"})
        print(f"  saved deployment artifact: results/ridge_with_clinical.joblib")

    # ------------------------------------------------------------ report ----
    report = render_report(metrics, table)
    (config.DOCS_DIR / "evaluation_report.md").write_text(report)
    print(f"  wrote docs/evaluation_report.md and results/metrics.json")
    return 0


def _fmt(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer (avoids a tabulate dependency)."""
    def cell(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "n/a"
        return f"{v:g}" if isinstance(v, float) else str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def render_report(metrics: dict, table: pd.DataFrame) -> str:
    split = metrics["data"]
    models = metrics["models"]
    base_p = models["persistence"]["chronological"]
    base_m = models["personal_mean"]["chronological"]
    ridge_c = models["ridge_with_clinical"]["chronological"]
    ridge_w = models["ridge_wearable_only"]["chronological"]
    ridge_b = models["ridge_with_baseline"]["chronological"]

    def better(a, b) -> str:
        if a is None or b is None:
            return "n/a"
        d = b - a
        return f"{d:+.3f} MAE ({'improvement' if d > 0 else 'no improvement'})"

    hgbr = metrics.get("hgbr_adoption", {})
    lines = [
        "# Biosphere — Evaluation Report",
        "",
        f"_Generated by `scripts/run_evaluation.py` on the shipped synthetic demo "
        f"dataset. Every number below was computed from actual model predictions on "
        f"held-out rows; nothing is estimated or copied from another run._",
        "",
        "> **All data are synthetic.** Results describe simulator behavior only and",
        "> carry no evidence about real patients, clinical validity, or medical",
        "> performance. See `docs/dataset_card.md` for the label assumptions.",
        "",
        "## Data & split",
        "",
        f"- Source: `{split['dir']}` (deterministic generator, "
        f"seed `{config.RANDOM_SEED}`)",
        f"- Usable prediction rows: **{split['n_usable_rows']}** across "
        f"**{split['n_patients']}** patients",
        f"- Chronological split: **{split['n_train_rows']} train rows / "
        f"{split['n_test_rows']} test rows** "
        f"({split['n_train_patients']} train patients, "
        f"{split['n_test_patients']} test patients), "
        f"test = last {split['test_fraction']:.0%} of each patient's days",
        f"- Held-out-patient experiment: every {split['heldout_modulus']}th patient "
        f"excluded from training entirely",
        "",
        "## Results (chronological held-out test rows)",
        "",
        _md_table(table),
        "",
        "## Reading the numbers honestly",
        "",
        f"- **Persistence baseline** MAE {base_p['mae']:.3f} "
        f"(n={base_p['n']}); **personal rolling mean** "
        f"{base_m['mae']:.3f} (n={base_m['n']}).",
        f"- **Ridge wearable-only** {ridge_w['mae']:.3f} vs persistence: "
        f"{better(ridge_w['mae'], base_p['mae'])}.",
        f"- **Ridge + personal baseline features** {ridge_b['mae']:.3f} vs "
        f"wearable-only: {better(ridge_b['mae'], ridge_w['mae'])}.",
        f"- **Ridge + clinical context** {ridge_c['mae']:.3f} vs "
        f"wearable+baseline: {better(ridge_c['mae'], ridge_b['mae'])}.",
        f"- Held-out patients (cold start): ridge+clinical MAE "
        f"{_fmt(models['ridge_with_clinical']['heldout_patients'].get('mae'))} "
        f"over {models['ridge_with_clinical']['heldout_patients'].get('n', 0)} rows "
        f"from {models['ridge_with_clinical']['heldout_patients'].get('n_patients', 0)} "
        f"unseen patients.",
        f"- Missing-data behavior: "
        f"{json.dumps(models['ridge_with_clinical']['missing_subsets'])}.",
        "",
    ]
    if hgbr:
        if hgbr.get("adopted"):
            lines.append(
                f"- HistGradientBoosting comparison: test MAE "
                f"{hgbr['hgbr_test_mae']:.3f} vs Ridge {hgbr['ridge_test_mae']:.3f} — "
                f"improvement observed; reported alongside the linear model.")
        else:
            reason = hgbr.get("note", "")
            tail = f" {reason}." if reason else "."
            lines.append(
                f"- HistGradientBoosting comparison: **not adopted** "
                f"(test MAE {_fmt(hgbr.get('hgbr_test_mae'))} vs Ridge "
                f"{_fmt(hgbr.get('ridge_test_mae'))}){tail} The simpler "
                f"regularized linear model is kept as the primary model.")
    lines += [
        "",
        "## Uncertainty proxy",
        "",
        f"- Ridge+clinical training residual std: "
        f"{_fmt(models['ridge_with_clinical']['train_residual_stats'].get('std'))} "
        f"(n={models['ridge_with_clinical']['train_residual_stats'].get('n')}).",
        "- The dashboard shows ±1.96σ bands built from this residual spread.",
        "  Following BioTwin's wording discipline, these are **error visualizations,",
        "  not calibrated prediction intervals**.",
        "",
        "## Threats to validity",
        "",
        "- Labels come from the simulator's documented equation, so 'improvement'",
        "  means the model recovers structure the simulator planted — useful for",
        "  engineering validation, not evidence of medical validity.",
        "- Clinical features help here because the simulator deliberately couples",
        "  condition burden to recovery. That coupling is an assumption, and the",
        "  ablation table would honestly show it if the coupling were removed.",
        "- Per-patient chronological windows overlap between train and test days",
        "  (different days, same patient); the held-out-patient experiment partially",
        "  addresses person-level generalization.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/generate_demo_data.py",
        "python scripts/run_evaluation.py",
        "python scripts/smoke_test.py",
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
