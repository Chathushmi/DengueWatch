"""DengueWatch Sri Lanka - Streamlit front-end for the EBM dengue forecaster.

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    ALL_FEATURES,
    CATEGORICAL,
    DISTRICTS,
    FEATURE_LABELS,
    load_raw,
    recompute_derived,
)
from src.paths import ARTIFACTS, FIGURES  # noqa: E402

st.set_page_config(page_title="DengueWatch Sri Lanka", page_icon="🦟", layout="wide")


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS / "ebm_model.pkl")


@st.cache_data
def load_tables():
    feats = pd.read_csv(ARTIFACTS / "features.csv", parse_dates=["week_start_date"])
    preds = pd.read_csv(ARTIFACTS / "predictions.csv", parse_dates=["week_start_date"])
    raw = load_raw()
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    info = json.loads((ARTIFACTS / "training_info.json").read_text())
    expl = json.loads((ARTIFACTS / "explanations.json").read_text())
    imp = pd.read_json(ARTIFACTS / "ebm_importance.json")
    return feats, preds, raw, metrics, info, expl, imp


missing = [p for p in ("ebm_model.pkl", "features.csv", "metrics.json") if not (ARTIFACTS / p).exists()]
if missing:
    st.error(f"Artifacts missing ({', '.join(missing)}). Run `python scripts/run_all.py` first.")
    st.stop()

ebm = load_model()
feats, preds, raw, metrics, info, expl, imp = load_tables()
thresholds = info["alert_thresholds"]


def label(term: str) -> str:
    if " & " in term:
        return " x ".join(FEATURE_LABELS.get(t, t) for t in term.split(" & "))
    return FEATURE_LABELS.get(term, term)


def forecast(row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Predicted next-week cases and per-feature contributions for a single row."""
    growth = float(ebm.predict(row[ALL_FEATURES])[0])
    cases = float(np.expm1(row["base_log"].iloc[0] + growth))
    d = ebm.explain_local(row[ALL_FEATURES]).data(0)
    contrib = pd.DataFrame({"term": d["names"], "contribution": d["scores"], "value": d["values"]})
    contrib["feature"] = contrib.term.map(label)
    contrib["effect_pct"] = (np.exp(contrib.contribution) - 1) * 100
    return max(cases, 0.0), contrib


def contribution_chart(contrib: pd.DataFrame, n: int = 10, title: str = "") -> go.Figure:
    top = contrib.reindex(contrib.contribution.abs().sort_values(ascending=False).index).head(n).iloc[::-1]
    colors = np.where(top.contribution > 0, "#d62728", "#1f77b4")
    hover = [f"value = {v}<br>effect = {p:+.1f}%" for v, p in zip(top.value, top.effect_pct)]
    fig = go.Figure(go.Bar(x=top.effect_pct, y=top.feature, orientation="h", marker_color=colors,
                           hovertext=hover, hoverinfo="text"))
    fig.update_layout(title=title, xaxis_title="Effect on next week's cases (%)  -  red raises, blue lowers",
                      height=420, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def risk_badge(cases: float, district: str) -> None:
    thr = thresholds[district]
    if cases > thr:
        st.error(f"HIGH RISK  -  forecast {cases:.0f} cases is above the {district} alert level of {thr:.0f}")
    else:
        st.success(f"NORMAL  -  forecast {cases:.0f} cases is below the {district} alert level of {thr:.0f}")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🦟 DengueWatch Sri Lanka")
st.sidebar.caption("Next-week dengue forecasts with an Explainable Boosting Machine")
page = st.sidebar.radio("Go to", ["Overview", "Forecast & What-if", "Explain the model", "Model performance"])
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Epidemiology Unit, Ministry of Health Sri Lanka + NASA POWER climate (2006-2025). "
    "Decision-support prototype, not a medical tool."
)

# ---------------------------------------------------------------- pages
if page == "Overview":
    st.title("Dengue in Sri Lanka - weekly cases and climate")
    st.write(
        "Dengue is spread by *Aedes* mosquitoes that breed in rain-filled containers. "
        "This app forecasts **next week's dengue cases** for Colombo, Gampaha and Kandy and "
        "explains *why* the model expects cases to go up or down."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weekly records", f"{len(raw):,}")
    c2.metric("Years covered", f"{raw.week_start_date.dt.year.min()} - {raw.week_start_date.dt.year.max()}")
    c3.metric("Total cases", f"{raw.cases.sum():,}")
    c4.metric("Peak week", f"{raw.cases.max():,} cases")

    district = st.selectbox("District", DISTRICTS)
    g = raw[raw.district == district]
    fig = go.Figure()
    fig.add_bar(x=g.week_start_date, y=g.rainfall_mm, name="Rainfall (mm)", yaxis="y2",
                marker_color="rgba(31,119,180,0.35)")
    fig.add_scatter(x=g.week_start_date, y=g.cases, name="Dengue cases", line=dict(color="#d62728"))
    fig.update_layout(height=430, yaxis=dict(title="Weekly cases"),
                      yaxis2=dict(title="Weekly rainfall (mm)", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.1), margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    season = raw.groupby(["district", "week"]).cases.mean().reset_index()
    col1.plotly_chart(px.line(season, x="week", y="cases", color="district",
                              title="Average cases by week of year",
                              labels={"week": "Week of year", "cases": "Mean cases"}), width="stretch")
    yearly = raw.assign(year=raw.week_start_date.dt.year).groupby(["year", "district"]).cases.sum().reset_index()
    col2.plotly_chart(px.bar(yearly, x="year", y="cases", color="district", title="Cases per year"),
                      width="stretch")
    st.info("2017 was Sri Lanka's worst dengue epidemic on record. 2020-2021 are lower partly because "
            "COVID-19 disrupted reporting.")

elif page == "Forecast & What-if":
    st.title("Forecast next week's dengue cases")
    c1, c2 = st.columns([1, 2])
    district = c1.selectbox("District", DISTRICTS)
    options = feats[feats.district == district].sort_values("week_start_date")
    dates = options.week_start_date.dt.date.tolist()
    default = len(dates) - 1
    week = c2.select_slider("Week to forecast (start date)", options=dates, value=dates[default])
    row = options[options.week_start_date.dt.date == week].copy()

    pred, contrib = forecast(row)
    actual = int(row.target_cases.iloc[0])
    m1, m2, m3 = st.columns(3)
    m1.metric("Cases last week (known)", int(row.cases_lag1.iloc[0]))
    m2.metric("Forecast for this week", f"{pred:.0f}", delta=f"{pred - row.cases_lag1.iloc[0]:+.0f} vs last week",
              delta_color="inverse")
    m3.metric("Actually reported", actual)
    risk_badge(pred, district)
    st.plotly_chart(contribution_chart(contrib, title="Why this forecast? (EBM local explanation)"),
                    width="stretch")

    st.subheader("What-if: change the weather and see the forecast move")
    st.caption("Adjust the recent weather or last week's cases. The model recomputes the forecast and its explanation.")
    w1, w2, w3, w4 = st.columns(4)
    rain_pct = w1.slider("Rainfall in the last 12 weeks (%)", 0, 300, 100, 10)
    temp_shift = w2.slider("Temperature change (°C)", -2.0, 2.0, 0.0, 0.25)
    hum_shift = w3.slider("Humidity change (% points)", -10, 10, 0, 1)
    cases_now = w4.number_input("Cases last week", 0, 3000, int(row.cases_lag1.iloc[0]), 5)

    what = row.copy()
    for k in range(1, 13):
        what[f"rain_lag{k}"] = row[f"rain_lag{k}"] * rain_pct / 100
    for k in (1, 4, 8, 12):
        what[f"temp_lag{k}"] = row[f"temp_lag{k}"] + temp_shift
        what[f"humidity_lag{k}"] = (row[f"humidity_lag{k}"] + hum_shift).clip(0, 100)
    what["cases_lag1"] = cases_now
    what = recompute_derived(what)
    pred_w, contrib_w = forecast(what)

    a, b = st.columns(2)
    a.metric("Original forecast", f"{pred:.0f} cases")
    b.metric("What-if forecast", f"{pred_w:.0f} cases", delta=f"{pred_w - pred:+.0f}", delta_color="inverse")
    risk_badge(pred_w, district)
    with st.expander("What-if explanation"):
        st.plotly_chart(contribution_chart(contrib_w, title="Explanation for the what-if scenario"),
                        width="stretch")
    with st.expander("Input values used"):
        show = what[ALL_FEATURES].T.rename(columns={what.index[0]: "value"})
        show.index = [label(i) for i in show.index]
        st.dataframe(show.astype(str), width="stretch")

elif page == "Explain the model":
    st.title("What has the model learned?")
    st.write(
        "An **Explainable Boosting Machine** adds up one learned curve per feature. Each curve below *is* the "
        "model - nothing is approximated. Positive values push next week's cases up."
    )
    top = imp.head(15).iloc[::-1]
    st.plotly_chart(px.bar(top, x="importance", y="label", orientation="h", title="Global feature importance",
                           labels={"importance": "Mean |contribution|", "label": ""}, height=480),
                    width="stretch")

    mains = [t for t in imp.term if " & " not in t]
    term = st.selectbox("Show the learned curve for", mains, format_func=label)
    d = ebm.explain_global().data(list(ebm.term_names_).index(term))
    scores = np.asarray(d["scores"])
    if term in CATEGORICAL:
        fig = px.bar(x=d["names"], y=(np.exp(scores) - 1) * 100, labels={"x": label(term), "y": "Effect (%)"})
    else:
        edges = np.asarray(d["names"], dtype=float)[:-1]
        fig = go.Figure()
        fig.add_scatter(x=edges, y=(np.exp(d["upper_bounds"]) - 1) * 100, line=dict(width=0), showlegend=False)
        fig.add_scatter(x=edges, y=(np.exp(d["lower_bounds"]) - 1) * 100, line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(214,39,40,0.15)", showlegend=False)
        fig.add_scatter(x=edges, y=(np.exp(scores) - 1) * 100, line=dict(color="#d62728", shape="hv"),
                        name="Effect")
        fig.update_layout(xaxis_title=label(term), yaxis_title="Effect on next week's cases (%)")
    fig.add_hline(y=0, line_color="black")
    fig.update_layout(height=380, title=f"Shape function: {label(term)}")
    st.plotly_chart(fig, width="stretch")

    tabs = st.tabs(["SHAP", "LIME", "Partial dependence", "Domain check"])
    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.image(str(FIGURES / "shap_bar.png"))
        c2.image(str(FIGURES / "shap_beeswarm.png"))
        ag = expl["agreement"]
        st.write(f"SHAP and the EBM agree on **{ag['overlap_top10']} of the top 10** features "
                 f"(Spearman rank correlation {ag['spearman_rank_corr']:.2f}).")
    with tabs[1]:
        c1, c2 = st.columns(2)
        c1.image(str(FIGURES / "lime_rising.png"))
        c2.image(str(FIGURES / "lime_falling.png"))
    with tabs[2]:
        st.image(str(FIGURES / "pdp_ice.png"))
    with tabs[3]:
        dc = expl["domain_check"]
        st.image(str(FIGURES / "domain_rain_lag.png"))
        st.write(
            f"Dengue research says cases follow rain by about **6-10 weeks** (mosquito breeding + virus "
            f"incubation). In this data, rainfall correlates most with cases **{dc['peak_raw_corr_lag']} weeks "
            f"later**, and the model's most important rainfall lag is **{dc['most_important_rain_lag']} weeks**."
        )

else:
    st.title("How good are the forecasts?")
    pct = metrics.get("split_percentages", {})
    split_note = (
        f"Split: train {pct.get('train', '?')}% / validation {pct.get('val', '?')}% / "
        f"test {pct.get('test', '?')}% (chronological)."
        if pct
        else "Chronological train / validation / test split."
    )
    st.caption(
        f"Test period {info['split_ranges']['test'][0]} to {info['split_ranges']['test'][1]} "
        f"({metrics['n_test']} district-weeks). {split_note}"
    )
    val_r2 = metrics["validation_ebm"]["R2"]
    fit_r2 = metrics.get("trainval_fit_ebm", {}).get("R2")
    m0, m1, m2 = st.columns(3)
    m0.metric("Test R² (EBM)", f"{metrics['test_overall']['ebm']['R2']:.2f}")
    m1.metric("Validation R² (EBM)", f"{val_r2:.2f}", help="Model trained only on earlier train years, scored on validation slice.")
    if fit_r2 is not None:
        m2.metric("Train+val fit R²", f"{fit_r2:.2f}", help="Final model refit on train+validation; shows how well curves fit the data used to train.")
    t = metrics["test_overall"]["ebm"]
    e0, e1, e2, e3 = st.columns(4)
    e0.metric("Test MAE", f"{t['MAE']:.1f}", help="Mean absolute error, cases per week.")
    e1.metric("Test MSE", f"{t['MSE']:,.0f}", help="Mean squared error (cases squared); punishes big misses.")
    e2.metric("Test RMSE", f"{t['RMSE']:.1f}", help="Square root of MSE, cases per week.")
    e3.metric("Median error", f"{t['MedAE']:.1f}", help="Error in a typical week, cases per week.")
    labels = metrics["model_labels"]
    reg_cols = ["MAE", "MSE", "RMSE", "MedAE", "WAPE", "sMAPE", "bias", "MASE", "R2", "pearson_r"]
    table = pd.DataFrame(metrics["test_overall"]).T.rename(index=labels)[reg_cols]
    alert_cols = ["precision", "recall", "specificity", "F1", "balanced_accuracy", "MCC", "ROC_AUC", "PR_AUC"]
    alert = pd.DataFrame(metrics["test_alert"]).T.rename(index=labels)[alert_cols]
    st.subheader("Case-count error (test)")
    st.dataframe(table.style.format({c: "{:,.0f}" if c == "MSE" else "{:.2f}" for c in reg_cols}), width="stretch")
    st.caption("WAPE and sMAPE are fractions of reported cases; bias < 0 means under-forecasting; "
               "MASE < 1 beats the naive 'same as last week' error from training.")
    st.subheader("Outbreak alert (test)")
    st.dataframe(alert.style.format("{:.3f}"), width="stretch")
    dr = metrics["test_direction"]["ebm"]
    st.write(
        f"The EBM matches the strong 'same as last week' baseline on error, and it also predicts **whether cases "
        f"will rise or fall** correctly {dr['direction_accuracy']:.0%} of the time "
        f"({dr['direction_accuracy_big_moves']:.0%} for large moves) - something the baseline cannot do."
    )
    district = st.selectbox("District", DISTRICTS)
    g = preds[(preds.split == "test") & (preds.district == district)]
    fig = go.Figure()
    fig.add_scatter(x=g.week_start_date, y=g.target_cases, name="Actual", line=dict(color="black"))
    fig.add_scatter(x=g.week_start_date, y=g.ebm, name="EBM forecast", line=dict(color="#d62728"))
    fig.add_hline(y=thresholds[district], line_dash="dot", line_color="orange", annotation_text="Alert level")
    fig.update_layout(height=420, yaxis_title="Weekly cases", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    c1.image(str(FIGURES / "model_comparison.png"))
    c2.image(str(FIGURES / "alert_confusion.png"))
