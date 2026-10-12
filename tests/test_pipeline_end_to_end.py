"""End-to-end pipeline test: the exact data path the dashboard runs at startup.

This is the 'dashboard startup critical path' test — load -> validate ->
features -> all models -> evaluation -> next-day prediction -> scenario —
executed on the small cohort, asserting every stage's contract.
"""
from __future__ import annotations

import config
from src import evaluation as EV
from src import models as M
from src import scenarios as S
from src import validation as V
from src.features import (build_feature_frame, build_next_day_rows,
                          merge_streams, usable_for_training)


def test_app_critical_path(loaded_demo, small_clinical):
    wearable = loaded_demo["wearable"]
    diary = loaded_demo["diary"]

    # 1. validation gates
    w_rep = V.validate_wearable(wearable)
    d_rep = V.validate_diary(diary)
    V.require_passed(w_rep, "wearable")
    V.require_passed(d_rep, "diary")
    V.validate_alignment(wearable, diary, small_clinical)

    # 2. features
    daily = merge_streams(wearable, diary)
    frame = build_feature_frame(daily, clinical=small_clinical)
    usable = usable_for_training(frame)
    assert len(usable) > 100

    # 3. all models fit and evaluate
    results = EV.run_full_evaluation(frame, include_hgbr=False)
    assert results["split"]["n_test_rows"] > 30
    best = min(results["models"].values(),
               key=lambda r: r.chronological["mae"] or 99)
    assert best.chronological["mae"] < 2.0, "sanity bound: synthetic labels"

    # 4. next-day prediction path (what the Prediction tab does)
    next_rows = build_next_day_rows(daily, clinical=small_clinical)
    assert len(next_rows) == 6
    artifacts = results["artifacts"]
    yhat = M.predict(artifacts["ridge_with_clinical"], next_rows)
    assert len(yhat) == 6 and yhat.min() >= 1 and yhat.max() <= 5

    # 5. scenario path (what the What-If tab does)
    train, _test = EV.make_train_test(usable)
    train_range = S.training_sleep_range(train)
    out = S.run_sleep_scenarios(artifacts["ridge_with_clinical"],
                                next_rows.iloc[0], [5.0, 9.0], train_range,
                                resid_stats=results["models"]
                                ["ridge_with_clinical"].train_residual_stats)
    assert len(out) == 2
    assert out["predicted_rating"].between(1, 5).all()

    # 6. demo-data disclaimer contract
    assert "synthetic" in config.SYNTHETIC_DISCLAIMER.lower()
