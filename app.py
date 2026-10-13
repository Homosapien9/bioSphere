"""BioSphere — personal recovery exploration tool (MVP).

Run: streamlit run app.py

Every number displayed is computed from the loaded data or a fitted model
at render time. No metric is hard-coded and no network call is made.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config                                            # noqa: E402
from src import data_generation as DG                    # noqa: E402
from src import evaluation as EV                         # noqa: E402
from src import features as FE                           # noqa: E402
from src import models as M                              # noqa: E402
from src import pipeline as PL                           # noqa: E402
from src import personalization as PZ                    # noqa: E402
from src import scenarios as SC                          # noqa: E402
from src import uncertainty as U                         # noqa: E402
from src import validation as V                          # noqa: E402

st.set_page_config(page_title="BioSphere", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- styling --
st.markdown("""
<style>
  .bs-card{background:#ffffff;border:1px solid #E5E7EB;border-radius:12px;
           padding:14px 18px;}
  .bs-big{font-size:2.1rem;font-weight:700;line-height:1.15;color:#111827;}
  .bs-label{font-size:0.75rem;font-weight:600;letter-spacing:.05em;
            text-transform:uppercase;color:#6B7280;}
  .bs-muted{color:#6B7280;font-size:0.86rem;}
  .bs-note{background:#FFFBEB;border:1px solid #FDE68A;color:#92400E;
           border-radius:10px;padding:10px 14px;font-size:0.9rem;}
  .bs-chip{display:inline-block;padding:2px 10px;border-radius:999px;
           font-size:0.72rem;font-weight:600;margin-right:6px;}
  .bs-chip-model{background:#EFF6FF;color:#1D4ED8;border:1px solid #BFDBFE;}
  .bs-chip-obs{background:#ECFDF5;color:#047857;border:1px solid #A7F3D0;}
</style>""", unsafe_allow_html=True)

METRIC_LABELS = {
    "sleep_duration_hours": "Sleep duration",
    "hrv_rmssd": "Heart-rate variability",
    "resting_hr": "Resting heart rate",
    "steps": "Steps",
}
METRIC_UNITS = {"sleep_duration_hours": "h", "hrv_rmssd": "ms",
                "resting_hr": "bpm", "steps": ""}

# ------------------------------------------------------------ cached state --
@st.cache_data(show_spinner="Loading and validating data…")
def load_data(data_dir: str) -> dict:
    data = PL.load_inputs(data_dir)
    data["w_val"] = V.validate_wearable(data["wearable"])
    data["d_val"] = V.validate_diary(data["diary"])
    data["a_val"] = V.validate_alignment(data["wearable"], data["diary"],
                                         data["clinical"])
    data["frame"], data["next_rows"] = FE.build_frames(
        data["daily"], clinical=data["clinical"])
    return data


@st.cache_resource(show_spinner="Testing the models on held-out data…")
def run_cached_evaluation(data_dir: str, sig: float) -> dict:
    data = load_data(data_dir)
    return EV.run_full_evaluation(data["frame"], include_hgbr=True)


@st.cache_resource(show_spinner="Fitting the deployment model…")
def get_deployment_artifacts(data_dir: str, sig: float) -> dict:
    """Model shown in the Predict tab: refit on ALL usable rows (standard
    deployment practice). Accuracy numbers elsewhere come from the held-out
    evaluation, never from this refit."""
    data = load_data(data_dir)
    usable = FE.usable_for_training(data["frame"])
    art = M.fit_sklearn("ridge", "with_clinical", usable)
    pers = M.fit_baseline("persistence", usable)
    train_pred = M.predict(art, usable)
    resid = U.residual_stats(usable[FE.TARGET].to_numpy(dtype=float),
                             train_pred)
    return {"ridge_with_clinical": art, "persistence": pers,
            "n_refit_rows": int(len(usable)), "resid": resid}


def dir_signature(path: Path) -> float:
    times = [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
    return max(times) if times else 0.0


def display_dir(path_str: str) -> str:
    try:
        return str(Path(path_str).resolve().relative_to(ROOT))
    except ValueError:
        return path_str


def plot(fig: go.Figure) -> None:
    """Version-tolerant full-width plotly rendering."""
    try:
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})


# ------------------------------------------------------- naming / explain --
def human_feature_name(f: str) -> str:
    base_map = {
        "sleep_duration_hours": "Sleep duration",
        "sleep_onset_hour": "Sleep onset time",
        "resting_hr": "Resting heart rate",
        "hrv_rmssd": "Heart-rate variability",
        "steps": "Steps",
        "rating": "Recovery rating",
    }
    # longest suffixes first so 'rating_lag1_missing' beats 'rating_lag1'
    suffix_map = [
        ("_lag1_missing", "(was missing)"),
        ("_roll7_mean", "(7-day average)"),
        ("_hist_mean", "(personal average so far)"),
        ("_dev_lag1", "(latest vs personal average)"),
        ("_lag2", "(two days ago)"),
        ("_lag1", "(most recent day)"),
    ]
    for suf, label in suffix_map:
        if f.endswith(suf):
            stem = f[:-len(suf)]
            if stem.startswith("rating"):
                stem = "rating"
            return f"{base_map.get(stem, stem.replace('_', ' '))} {label}".strip()
    special = {
        "rating_expanding_mean": "Recovery rating (personal average so far)",
        "rating_dev_lag1": "Latest rating vs personal average",
        "rating_lag1_missing": "Yesterday's rating (was missing)",
        "age": "Age", "bmi": "BMI", "gender_male": "Sex (male)",
        "systolic_bp": "Systolic blood pressure",
        "diastolic_bp": "Diastolic blood pressure",
        "clinical_resting_hr": "Clinical resting heart rate",
    }
    return special.get(f, f.replace("_", " "))


def ridge_contributions(artifact: dict, row: pd.Series,
                        top_k: int = 6) -> pd.DataFrame | None:
    """Local explanation: each factor's contribution to this estimate, in
    rating points (standardized value x coefficient). Returns None if the
    decomposition is unavailable for this model."""
    try:
        pipe = artifact["model"]
        feats = artifact["features"]
        X = pd.DataFrame([[row.get(f, np.nan) for f in feats]], columns=feats)
        Xt = pipe.named_steps["scale"].transform(
            pipe.named_steps["impute"].transform(X))
        coef = pipe.named_steps["ridge"].coef_
        contrib = Xt[0] * coef
        order = np.argsort(-np.abs(contrib))[:top_k]
        return pd.DataFrame({
            "Factor": [human_feature_name(feats[i]) for i in order],
            "Effect": ["raises" if contrib[i] > 0 else "lowers"
                       for i in order],
            "Impact (points)": [round(float(contrib[i]), 3) for i in order],
        })
    except Exception:
        return None


# ------------------------------------------------------------------ charts --
def rating_history_chart(pg: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pg["date"], y=pg[FE.TARGET], mode="lines+markers",
        name="recorded rating", connectgaps=False,
        line=dict(color="#2563EB", width=2), marker=dict(size=5),
        hovertemplate="%{x|%a %b %d}: %{y:.0f} / 5<extra></extra>"))
    med = pg[FE.TARGET].median()
    if pd.notna(med):
        fig.add_hline(y=float(med), line_dash="dash", line_color="#9CA3AF",
                      annotation_text=f"your typical rating (median {med:.0f})",
                      annotation_position="top left")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis=dict(range=[0.5, 5.5], dtick=1),
                      showlegend=False)
    return fig


def metric_chart(pg: pd.DataFrame, metric: str,
                 median: float | None) -> go.Figure:
    unit = METRIC_UNITS.get(metric, "")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pg["date"], y=pg[metric], mode="lines+markers",
        name="recorded", connectgaps=False,
        line=dict(color="#2563EB", width=2), marker=dict(size=4),
        hovertemplate="%{x|%a %b %d}: %{y:.1f} " + unit + "<extra></extra>"))
    tail = pg.dropna(subset=[metric]).tail(config.RECENT_WINDOW_DAYS)
    fig.add_trace(go.Scatter(
        x=tail["date"], y=tail[metric], mode="markers",
        name=f"last {config.RECENT_WINDOW_DAYS} days",
        marker=dict(color="#DC2626", size=9, symbol="diamond")))
    if median is not None and pd.notna(median):
        fig.add_hline(y=float(median), line_dash="dash", line_color="#9CA3AF",
                      annotation_text=f"your typical {median:g} {unit}",
                      annotation_position="top left")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                      title=dict(text=METRIC_LABELS.get(metric, metric),
                                 font=dict(size=14)),
                      legend=dict(orientation="h", y=1.15, x=0))
    return fig


def pred_vs_actual_chart(sub: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(sub["date"]), y=sub["y_true"], mode="lines+markers",
        name="actual rating", line=dict(color="#111827", width=2),
        marker=dict(size=5)))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(sub["date"]), y=sub["y_pred"], mode="lines+markers",
        name="model estimate", line=dict(color="#2563EB", width=2, dash="dot"),
        marker=dict(size=5)))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis=dict(range=[0.5, 5.5], dtick=1),
                      legend=dict(orientation="h", y=1.18, x=0))
    return fig


def mae_bar_chart(table: pd.DataFrame, order: list[str],
                  labels: dict[str, str]) -> go.Figure:
    t = table.copy()
    t["__o"] = t["model"].map({m: i for i, m in enumerate(order)})
    t = t.sort_values("__o")
    colors = ["#9CA3AF", "#9CA3AF", "#93C5FD", "#2563EB", "#1D4ED8", "#A78BFA"]
    fig = go.Figure()
    fig.add_bar(x=[labels.get(m, m) for m in t["model"]], y=t["mae"],
                marker_color=colors[:len(t)],
                text=[f"{v:.2f}" if pd.notna(v) else "n/a" for v in t["mae"]],
                textposition="outside")
    pers = t[t["model"] == "persistence"]["mae"]
    if len(pers) and pd.notna(pers.iloc[0]):
        fig.add_hline(y=float(pers.iloc[0]), line_dash="dash",
                      line_color="#9CA3AF",
                      annotation_text="simple-guess baseline")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="average error (rating points, lower is better)")
    fig.update_xaxes(tickfont=dict(size=10))
    return fig


# -------------------------------------------------------------------- main --
def main() -> None:
    st.title("BioSphere")
    st.caption("A personal recovery exploration tool. It estimates tomorrow's "
               "self-reported recovery rating (1–5) from wearable data — "
               "built and tested on simulated data only.")
    st.markdown(
        '<div class="bs-note"><b>Simulated data.</b> Every record here comes '
        'from a seeded simulator. Estimates are model outputs, not '
        'measurements, not medical advice, and not proof of any health '
        'effect.</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------ sidebar --
    with st.sidebar:
        st.header("Setup")
        shipped = config.DEMO_DATA_DIR
        generated = config.GENERATED_DATA_DIR
        options = {"Shipped demo dataset": shipped}
        if (generated / config.WEARABLE_CSV_NAME).exists():
            options["Regenerated dataset"] = generated
        source_label = st.selectbox("Data source", list(options), index=0)
        data_dir = Path(options[source_label])
        sig = dir_signature(data_dir)
        if st.button("Regenerate simulated cohort", use_container_width=True,
                     help="Writes a fresh deterministic cohort (same seed) "
                          "to data/generated and switches to it."):
            with st.spinner("Generating…"):
                cohort = DG.generate_cohort()
                DG.write_cohort(cohort, generated)
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        try:
            data = load_data(str(data_dir))
        except Exception as exc:
            st.error(f"Could not load data from {data_dir}: {exc}")
            st.stop()
        patient_ids = sorted(data["wearable"]["patient_id"].unique())
        patient = st.selectbox("Person (simulated)", patient_ids)
        st.divider()
        st.caption(f"Folder: `{display_dir(data['dir'])}` · "
                   f"seed `{config.RANDOM_SEED}`")

    try:
        results = run_cached_evaluation(str(data_dir), sig)
        deployment = get_deployment_artifacts(str(data_dir), sig)
    except Exception as exc:
        st.error(f"Evaluation failed and is being reported, not hidden: {exc}")
        st.stop()

    daily = data["daily"]
    pg = daily[daily["patient_id"] == patient].sort_values("date")
    baseline = PZ.PersonalBaseline().fit(
        daily[daily["patient_id"] == patient])

    tab_overview, tab_explore, tab_predict, tab_details = st.tabs(
        ["Overview", "Explore", "Predict", "Model details"])

    # ================================================================ Overview
    with tab_overview:
        if pg.empty:
            st.warning("No recorded data for this person.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown('<div class="bs-card">'
                            '<div class="bs-label">Days of data</div>'
                            f'<div class="bs-big">{len(pg)}</div></div>',
                            unsafe_allow_html=True)
            with c2:
                n_ratings = int(pg[FE.TARGET].notna().sum())
                st.markdown('<div class="bs-card">'
                            '<div class="bs-label">Ratings logged</div>'
                            f'<div class="bs-big">{n_ratings}</div></div>',
                            unsafe_allow_html=True)
            with c3:
                comp = np.mean([pg[m].notna().mean() for m in
                                config.WEARABLE_COLUMNS])
                st.markdown('<div class="bs-card">'
                            '<div class="bs-label">Sensor coverage</div>'
                            f'<div class="bs-big">{100 * comp:.0f}%</div>'
                            '</div>', unsafe_allow_html=True)
            with c4:
                st.markdown('<div class="bs-card">'
                            '<div class="bs-label">Period</div>'
                            f'<div class="bs-big" style="font-size:1.15rem">'
                            f'{pg["date"].min():%b %d} – '
                            f'{pg["date"].max():%b %d, %Y}</div></div>',
                            unsafe_allow_html=True)

            crow = data["clinical"][data["clinical"]["patient_id"] == patient]
            if not crow.empty:
                c = crow.iloc[0]
                bits = []
                if pd.notna(c.get("age")):
                    bits.append(f"age {int(c['age'])}")
                bits.append("male" if c.get("gender_male") else "female")
                if pd.notna(c.get("bmi")):
                    bits.append(f"BMI {c['bmi']:.1f}")
                st.caption("Simulated clinical record: " + " · ".join(bits))

            st.markdown("#### Your recovery ratings over time")
            plot(rating_history_chart(pg))
            med = pg[FE.TARGET].median()
            if pd.notna(med):
                lo, hi = pg[FE.TARGET].min(), pg[FE.TARGET].max()
                st.caption(f"Recorded ratings range from {lo:.0f} to {hi:.0f}; "
                           f"the dashed line is your typical rating "
                           f"(median {med:.0f}). Gaps are days you didn't log "
                           "a rating — they are never filled in.")

            st.markdown("#### Recording coverage by measurement")
            rows = []
            for m in config.WEARABLE_COLUMNS:
                present = int(pg[m].notna().sum())
                rows.append({
                    "Measurement": METRIC_LABELS.get(m, m),
                    "Days recorded": present,
                    "Days in period": len(pg),
                    "Coverage": f"{100 * present / len(pg):.0f}%",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # ================================================================= Explore
    with tab_explore:
        if pg.empty:
            st.warning("No recorded data for this person.")
        else:
            metric = st.radio(
                "Measurement", list(METRIC_LABELS),
                format_func=METRIC_LABELS.get, horizontal=True)
            max_days = max(14, len(pg))
            if len(pg) > 14:
                window = st.slider("Show last N days", 14, len(pg),
                                   len(pg), step=1)
            else:
                window = len(pg)
            view = pg.tail(window)
            s = baseline.summary(patient, metric) or {}
            plot(metric_chart(view, metric, s.get("median")))
            missing = int(pg[metric].isna().sum())
            if missing:
                st.caption(f"{missing} of {len(pg)} days have no recorded "
                           f"{METRIC_LABELS.get(metric, metric).lower()} — "
                           "shown as gaps in the line, never filled in.")

            st.markdown("#### Last few days vs your own typical values")
            rows = []
            for m in config.WEARABLE_COLUMNS:
                cmp_ = baseline.recent_comparison(daily, patient, m)
                if cmp_.get("status") != "ok":
                    continue
                z = cmp_.get("robust_z")
                if z is None:
                    status = "not enough variation to compare"
                elif abs(z) < 1:
                    status = "about your typical"
                elif z > 0:
                    status = "above your typical"
                else:
                    status = "below your typical"
                if not cmp_.get("sufficient", True):
                    status += " (short history — provisional)"
                unit = METRIC_UNITS.get(m, "")
                diff = None
                if cmp_.get("recent_mean") is not None \
                        and cmp_.get("baseline_median") is not None:
                    diff = round(cmp_["recent_mean"]
                                 - cmp_["baseline_median"], 1)
                rows.append({
                    "Measurement": METRIC_LABELS.get(m, m),
                    f"Last {cmp_.get('recent_k', config.RECENT_WINDOW_DAYS)}"
                    " days (avg)": f"{cmp_.get('recent_mean', 'n/a')} {unit}",
                    "Your typical (median)":
                        f"{cmp_.get('baseline_median', 'n/a')} {unit}",
                    "Difference": diff,
                    "Reading": status,
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             width="stretch")
                st.caption("“Above/below your typical” describes where recent "
                           "values sit relative to this person's own history. "
                           "It is not a diagnosis or a health signal.")

    # ================================================================= Predict
    with tab_predict:
        prow = data["next_rows"][data["next_rows"]["patient_id"] == patient]
        if prow.empty:
            st.info("Not enough recorded history yet to estimate tomorrow "
                    "for this person. Pick another person or regenerate the "
                    "dataset.")
        else:
            row = prow.iloc[0].copy()
            horizon = row["date"]
            st.markdown(f"#### Estimate for tomorrow — "
                        f"**{horizon:%A, %b %d, %Y}**")

            art = deployment["ridge_with_clinical"]
            X = pd.DataFrame([row[art["features"]].to_dict()],
                             columns=art["features"])
            yhat = float(M.predict(art, X)[0])
            resid = deployment["resid"]
            interval = U.gaussian_interval(yhat, resid)

            last_rating = row.get("rating_lag1")
            exp_mean = row.get("rating_expanding_mean")
            gmean = deployment["persistence"]["global_mean"]

            def _baseline_or_fallback(v):
                if pd.notna(v):
                    return float(np.clip(v, 1, 5)), False
                return float(np.clip(gmean, 1, 5)), True

            pers, pers_fb = _baseline_or_fallback(last_rating)
            pmean, pmean_fb = _baseline_or_fallback(exp_mean)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="bs-card">'
                            '<span class="bs-chip bs-chip-obs">SIMPLE GUESS'
                            '</span><div class="bs-muted">repeat last rating'
                            '</div><div class="bs-big">{:.2f}</div></div>'
                            .format(pers), unsafe_allow_html=True)
                st.caption("Tomorrow = the most recent rating you logged"
                           + (f" ({last_rating:g})" if not pers_fb
                              else f" — none logged, so the group average "
                                   f"{gmean:.2f} is used"))
            with c2:
                st.markdown('<div class="bs-card">'
                            '<span class="bs-chip bs-chip-obs">SIMPLE GUESS'
                            '</span><div class="bs-muted">your average so far'
                            '</div><div class="bs-big">{:.2f}</div></div>'
                            .format(pmean), unsafe_allow_html=True)
                st.caption("Tomorrow = your running average of past ratings"
                           + ("" if not pmean_fb
                              else " — not enough history, group average used"))
            with c3:
                st.markdown('<div class="bs-card">'
                            '<span class="bs-chip bs-chip-model">'
                            'MODEL ESTIMATE</span>'
                            '<div class="bs-muted">all your data</div>'
                            '<div class="bs-big">{:.2f}</div></div>'
                            .format(yhat), unsafe_allow_html=True)
                if interval:
                    st.caption(f"typical error band: "
                               f"[{interval[0]:.2f}, {interval[1]:.2f}] — "
                               "based on this model's past errors, not a "
                               "guarantee")
            if pd.isna(last_rating):
                st.info("No recent rating is available, so this estimate "
                        "relies on sensor data only and is less reliable.")

            st.caption("The estimate uses a model refit on all available data "
                       "(standard practice for a deployed predictor). The "
                       "accuracy numbers in **Model details** come from a "
                       "version trained only on earlier data.")

            with st.expander("Factors behind this estimate"):
                contrib = ridge_contributions(art, row)
                if contrib is None:
                    st.caption("A detailed factor breakdown is not available "
                               "for this model.")
                else:
                    st.dataframe(contrib, hide_index=True, width="stretch")
                    st.caption("Impact is each factor's contribution to this "
                               "estimate in rating points, relative to an "
                               "average day. Positive values push the estimate "
                               "up. This describes what the model used — not "
                               "what causes your recovery.")

            # ---- What-if ------------------------------------------------
            st.divider()
            st.markdown("#### What-if: change tonight's sleep")
            if pd.isna(row["sleep_duration_hours_lag1"]):
                st.info("Last night's sleep wasn't recorded for this person, "
                        "so the scenario has no starting point.")
            else:
                orig_sleep = float(row["sleep_duration_hours_lag1"])
                train_range = SC.training_sleep_range(
                    FE.usable_for_training(data["frame"]))
                default = float(np.clip(orig_sleep,
                                        *config.SCENARIO_SLEEP_RANGE))
                skey = f"sleep_slider_{patient}"
                if skey not in st.session_state:
                    st.session_state[skey] = default
                if st.button("Reset to recorded sleep"):
                    st.session_state[skey] = default
                    st.rerun()
                slider = st.slider(
                    "Hypothetical sleep tonight (hours)",
                    min_value=float(config.SCENARIO_SLEEP_RANGE[0]),
                    max_value=float(config.SCENARIO_SLEEP_RANGE[1]),
                    key=skey, step=0.25, format="%.2f h")
                both = SC.run_sleep_scenarios(
                    art, row, [orig_sleep, slider], train_range,
                    resid_stats=resid)
                base_pred = float(both.iloc[0]["predicted_rating"])
                scen_pred = float(both.iloc[1]["predicted_rating"])
                delta = scen_pred - base_pred

                c1, c2 = st.columns(2)
                with c1:
                    st.metric(f"Recorded sleep {orig_sleep:.1f} h → estimate",
                              f"{base_pred:.2f}")
                with c2:
                    st.metric(f"Hypothetical sleep {slider:.1f} h → estimate",
                              f"{scen_pred:.2f}",
                              delta=f"{delta:+.2f}", delta_color="off")
                if abs(delta) < 0.005:
                    st.caption("Changing sleep by this amount barely moves "
                               "the model's estimate for this person.")
                else:
                    st.caption(
                        f"Compared with your recorded {orig_sleep:.1f} h, "
                        f"sleeping {slider:.1f} h changes the model's "
                        f"estimate by {delta:+.2f} rating points.")
                if train_range and (slider < train_range[0]
                                    or slider > train_range[1]):
                    st.warning(f"{slider:.1f} h is outside the range of sleep "
                               f"seen in the training data "
                               f"({train_range[0]:.1f}–{train_range[1]:.1f} h). "
                               "Treat this as extrapolation.")
                st.caption("Hypothetical model output: one input changed, "
                           "everything else held fixed. This is not a "
                           "guaranteed physiological outcome and not evidence "
                           "of a causal effect.")

            # ---- Past performance for this person ------------------------
            st.divider()
            st.markdown("#### How the estimate would have done for this person")
            preds = results["predictions"]
            sub = pd.DataFrame()
            if isinstance(preds, pd.DataFrame) and not preds.empty:
                sub = preds[(preds["model"] == "ridge_with_clinical")
                            & (preds["patient_id"] == patient)]
            if sub.empty:
                st.caption("This person was kept out of the model's training "
                           "data entirely (or has no held-out days), so there "
                           "is no fair past-performance comparison to show.")
            else:
                plot(pred_vs_actual_chart(sub))
                errs = (sub["y_pred"] - sub["y_true"]).abs()
                st.caption(f"On {len(sub)} days the model was not trained on, "
                           f"its average miss for this person was "
                           f"{errs.mean():.2f} rating points.")

    # ============================================================ Model details
    with tab_details:
        st.markdown("#### How accurate is this model?")
        st.caption("Everything below was measured on data the model was not "
                   "trained on, using the shipped simulated dataset. These "
                   "numbers demonstrate pipeline behavior — they say nothing "
                   "about real patients.")

        table = EV.results_table(results)
        order = ["persistence", "personal_mean", "ridge_wearable_only",
                 "ridge_with_baseline", "ridge_with_clinical",
                 "hgbr_with_clinical"]
        labels = {
            "persistence": "Simple guess: repeat last rating",
            "personal_mean": "Simple guess: personal average",
            "ridge_wearable_only": "Model: sensor data only",
            "ridge_with_baseline": "Model: sensors + rating history",
            "ridge_with_clinical": "Model: sensors + history + clinical",
            "hgbr_with_clinical": "Comparison: gradient-boosted trees",
        }
        show = table[table["model"].isin(order)].copy()
        show["__o"] = show["model"].map({m: i for i, m in enumerate(order)})
        show = show.sort_values("__o").drop(columns="__o")
        show["Model"] = show["model"].map(labels)
        st.dataframe(show[["Model", "n", "mae", "rmse"]].rename(columns={
            "n": "Days tested", "mae": "Average error (MAE)",
            "rmse": "RMSE"}), hide_index=True, width="stretch")

        m_pers = results["models"].get("persistence", {}) \
            .chronological.get("mae")
        m_model = results["models"].get("ridge_with_clinical", {}) \
            .chronological.get("mae")
        if m_pers is not None and m_model is not None:
            if m_model < m_pers:
                st.success(
                    f"On days it never trained on, the full model's average "
                    f"error was **{m_model:.2f}** vs **{m_pers:.2f}** for the "
                    f"'repeat yesterday' guess — an improvement of "
                    f"{m_pers - m_model:.2f} rating points.")
            else:
                st.warning(
                    f"On days it never trained on, the full model "
                    f"(error {m_model:.2f}) did **not** beat the simple "
                    f"'repeat yesterday' guess ({m_pers:.2f}). We report "
                    "that as-is rather than hide it.")
        bt = results.get("bootstrap", {}) \
            .get("ridge_with_clinical_vs_persistence")
        if bt:
            st.caption(
                f"Uncertainty: re-sampling the {bt['n']} test days "
                f"({bt['n_boot']} draws) gives a 95% range of "
                f"[{bt['ci_low']:+.2f}, {bt['ci_high']:+.2f}] for the "
                f"improvement. Days within one person are correlated, so the "
                f"true uncertainty is larger.")

        plot(mae_bar_chart(show, order, labels))

        pp = results["predictions"]
        if isinstance(pp, pd.DataFrame) and not pp.empty:
            sub = pp[pp["model"] == "ridge_with_clinical"]
            if not sub.empty:
                fig = go.Figure()
                fig.add_scatter(x=[1, 5], y=[1, 5], mode="lines",
                                line=dict(dash="dash", color="#9CA3AF"),
                                name="perfect estimate")
                fig.add_scatter(x=sub["y_true"], y=sub["y_pred"],
                                mode="markers", opacity=0.55,
                                marker=dict(size=6, color="#2563EB"),
                                name="held-out days")
                fig.update_layout(height=320,
                                  xaxis_title="actual rating",
                                  yaxis_title="model estimate",
                                  yaxis=dict(range=[0.5, 5.5]),
                                  xaxis=dict(range=[0.5, 5.5]),
                                  margin=dict(l=10, r=10, t=20, b=10))
                plot(fig)

        st.markdown("#### Tested situations")
        split = results["split"]
        st.markdown(f"**1. Future days for the same people.** The last "
                    f"{split['test_fraction']:.0%} of each person's usable "
                    f"days ({split['n_test_rows']} days across "
                    f"{split['n_test_patients']} people) was held out; the "
                    f"model trained on the earlier "
                    f"{split['n_train_rows']} days.")
        ho_m = results["models"]["ridge_with_clinical"].heldout_patients
        ho_p = results["models"]["persistence"].heldout_patients
        st.markdown(f"**2. New people the model never trained on** "
                    f"(they still have their own logged history). "
                    f"{ho_m.get('n', 0)} days from {ho_m.get('n_patients', 0)} "
                    f"people: model error **{ho_m.get('mae', 'n/a')}** vs "
                    f"simple guess **{ho_p.get('mae', 'n/a')}**. "
                    f"⚠️ This is *not* a no-history cold start — the model "
                    f"still sees each person's own rating history.")
        nh = results.get("no_history", {})
        if nh.get("n", 0) > 0:
            st.markdown(f"**3. Days with no prior rating history** "
                        f"({nh['n']} days, {nh['n_patients']} people). "
                        f"Group-average guess error: "
                        f"**{nh.get('mae_global_mean', 'n/a')}**; "
                        f"sensor-only model error: "
                        f"**{nh.get('mae_wearable_only', 'n/a')}**. "
                        f"These days are mostly each person's first logged "
                        f"day, and the sensor-only model was trained only on "
                        f"days that had history — a distribution shift, so "
                        f"treat these numbers as indicative only.")
        else:
            st.markdown("**3. Days with no prior rating history.** "
                        "None occurred in this dataset.")

        st.markdown("#### Does each data stream help?")
        try:
            m_wo = results["models"]["ridge_wearable_only"] \
                .chronological["mae"]
            m_wb = results["models"]["ridge_with_baseline"] \
                .chronological["mae"]
            m_wc = results["models"]["ridge_with_clinical"] \
                .chronological["mae"]
            st.dataframe(pd.DataFrame([
                {"Comparison": "Sensor-only model vs simple guess",
                 "Change in average error": round(m_wo - m_pers, 3)},
                {"Comparison": "Adding rating history (vs sensor-only)",
                 "Change in average error": round(m_wb - m_wo, 3)},
                {"Comparison": "Adding clinical context (vs history)",
                 "Change in average error": round(m_wc - m_wb, 3)},
            ]), hide_index=True, width="stretch")
            st.caption("Negative = better. With only 24 simulated people, "
                       "differences smaller than about 0.05 points are "
                       "indistinguishable from noise.")
        except KeyError:
            st.caption("Ablation numbers unavailable.")

        hc = results.get("hgbr_comparison", {})
        if hc.get("error"):
            st.warning(f"The gradient-boosted comparison model failed to "
                       f"train and is reported instead of hidden: "
                       f"{hc['error']}")
        elif hc.get("available"):
            st.caption("The gradient-boosted tree model is included as a "
                       "reported comparison. It is never used to choose the "
                       "deployed model: Ridge was picked for "
                       "interpretability before any test results were seen.")
        else:
            st.caption("No gradient-boosted comparison available.")

        with st.expander("Technical details"):
            st.markdown(f"- Seed: `{config.RANDOM_SEED}`; generator: "
                        f"`src/data_generation.py`")
            st.markdown(f"- Features use only the last 1–"
                        f"{config.ROLLING_WINDOW} recorded days per person "
                        f"(shifted before any rolling/expanding statistic).")
            st.markdown(f"- Missing sensor values are imputed with the "
                        f"training-set median inside the model pipeline and "
                        f"flagged with missing-indicator features; displayed "
                        f"data is never filled in.")
            ms = results["models"]["ridge_with_clinical"].missing_subsets
            st.dataframe(pd.DataFrame([
                {"Subset": "last night fully recorded",
                 "n": ms["complete_last_night"]["n"],
                 "MAE": ms["complete_last_night"]["mae"]},
                {"Subset": "last night partially missing",
                 "n": ms["incomplete_last_night"]["n"],
                 "MAE": ms["incomplete_last_night"]["mae"]},
            ]), hide_index=True, width="stretch")
            pp_df = results["models"]["ridge_with_clinical"].per_patient
            if pp_df is not None and not pp_df.empty:
                fig = go.Figure()
                fig.add_bar(x=pp_df["patient_id"], y=pp_df["mae"],
                            marker_color="#2563EB")
                fig.update_layout(height=280, yaxis_title="average error",
                                  margin=dict(l=10, r=10, t=20, b=10))
                fig.update_xaxes(tickangle=45, tickfont=dict(size=9))
                plot(fig)
            st.markdown(f"- Load/validation report: "
                        f"wearable {data['w_val'].as_dict().get('status', '?')}"
                        f", diary {data['d_val'].as_dict().get('status', '?')}"
                        f", alignment "
                        f"{data['a_val'].as_dict().get('status', '?')}")

        st.markdown("#### Limitations")
        st.markdown(
            "- All data is simulated; the rating equation is an invented "
            "simulator assumption. Nothing here transfers to real patients.\n"
            "- 24 people is small; differences under ~0.05 error points are "
            "noise, and the bootstrap range understates uncertainty because "
            "days within a person are correlated.\n"
            "- Error bands are residual-based proxies, not calibrated "
            "intervals.\n"
            "- The what-if changes one input and holds the rest fixed — an "
            "association view, not physiology simulation and not a "
            "demonstrated causal effect.\n"
            "- Clinical context is a static snapshot; its value for new "
            "people is unproven at this scale.")


if __name__ == "__main__":
    main()
    
