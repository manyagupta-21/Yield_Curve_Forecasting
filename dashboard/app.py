"""
Yield Curve Forecasting Dashboard
====================================
Tab 1: Yield Curve Forecasting: single-maturity forecast-vs-actual, and the
         full curve snapshot across maturities on a chosen date, using each
         maturity's RMSE-best model.
Tab 2: Return / Pricing Engine: price/roll decomposition and directional
         accuracy from the zero-coupon bond return engine.

Run with:  streamlit run dashboard/app.py

Reads only pre-computed parquet files your notebooks already save — no
model refitting happens here, so the dashboard stays fast.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # so we can import the engine
from bond_return_engine import (
    YIELD_COLS, MATURITIES, run_return_engine, directional_accuracy,
)

# ----------------------------------------------------------------------
PROC = Path(r"D:\yield_curve_forecast\data\processed")
MATURITY_YEARS = dict(zip(YIELD_COLS, MATURITIES))

BEST_MODEL = {
    "DGS3MO": "DNS_VAR",
    "DGS1":   "ARIMA",
    "DGS2":   "VAR_diff",
    "DGS5":   "DNS_AR",
    "DGS10":  "DNS_AR",
}
DNS_MODELS = {"DNS_AR", "DNS_VAR"}
ALL_MODELS = ["RandomWalk", "ARIMA", "ARIMAX", "VAR_diff", "DNS_AR", "DNS_VAR"]

st.set_page_config(layout="wide", page_title="Yield Curve Dashboard")
st.title("US Treasury Yield Curve — Forecasting & Return Engine")


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_data
def load_actual():
    Y = pd.read_parquet(PROC / "yield_curve_data.parquet").set_index("observation_date")[YIELD_COLS]
    return Y.dropna()


@st.cache_data
def load_all_forecasts():
    """Combine 04_forecasting's and 05_dns's saved forecasts into one long
    frame: columns = model__maturity, same shape as predictions.parquet."""
    p3 = PROC / "predictions.parquet"
    p3b = PROC / "dns_predictions.parquet"
    frames = []
    if p3.exists():
        frames.append(pd.read_parquet(p3))
    if p3b.exists():
        frames.append(pd.read_parquet(p3b))
    if not frames:
        st.error(
            "No forecast files found. Run 04_forecasting's save cell "
            "(predictions.parquet) and 05_dns's save cell "
            "(dns_predictions.parquet) first."
        )
        st.stop()
    return pd.concat(frames, axis=1)


@st.cache_data
def load_actual_test():
    p = PROC / "actual.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return load_actual()  # fallback: full panel, caller slices test dates


def model_series(all_fc, model, maturity):
    col = f"{model}__{maturity}"
    if col not in all_fc.columns:
        return None
    return all_fc[col]


actual = load_actual()
all_fc = load_all_forecasts()
test_dates = all_fc.index

tab1, tab2 = st.tabs(["📈 Yield Curve Forecasting", "💰 Return / Pricing Engine"])

# ========================================================================
# TAB 1 — Yield Curve Forecasting
# ========================================================================
with tab1:
    sub1, sub2 = st.tabs(["Single Maturity", "Full Curve Snapshot"])

    # --- Single maturity: actual vs its RMSE-best model, over the test window ---
    with sub1:
        st.subheader("Forecast vs. Actual for a single maturity")
        maturity = st.selectbox("Maturity", YIELD_COLS, key="single_mat")
        best_model = BEST_MODEL[maturity]

        model_choice = st.radio(
            "Model", [f"Best (RMSE): {best_model}", "Compare all models"],
            horizontal=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual.loc[test_dates].index, y=actual.loc[test_dates, maturity],
            name="Actual", line=dict(color="black", width=2),
        ))

        if model_choice.startswith("Best"):
            s = model_series(all_fc, best_model, maturity)
            fig.add_trace(go.Scatter(x=test_dates, y=s, name=f"{best_model} (best)",
                                      line=dict(color="crimson", dash="dash")))
        else:
            palette = ["crimson", "steelblue", "seagreen", "darkorange", "purple", "grey", "teal"]
            for m, c in zip(ALL_MODELS, palette):
                s = model_series(all_fc, m, maturity)
                if s is not None:
                    fig.add_trace(go.Scatter(x=test_dates, y=s, name=m,
                                              line=dict(color=c, dash="dash"), opacity=0.7))

        fig.update_layout(height=480, xaxis_title="Date", yaxis_title="Yield (%)",
                           title=f"{maturity}: one-month-ahead forecast vs. actual")
        st.plotly_chart(fig, use_container_width=True)

        rmse = np.sqrt(np.nanmean((actual.loc[test_dates, maturity] -
                                    model_series(all_fc, best_model, maturity)) ** 2))
        st.metric(f"RMSE — {best_model} on {maturity}", f"{rmse:.4f}")

    # show actual + each maturity's best-model forecast for a particular date
    with sub2:
        st.subheader("Full Yield Curve Snapshot")
        snap_date = st.select_slider("Date", options=list(test_dates),
                                      value=test_dates[len(test_dates)//2],
                                      format_func=lambda d: d.strftime("%Y-%m"))

        mats = [MATURITY_YEARS[c] for c in YIELD_COLS]
        actual_curve = actual.loc[snap_date, YIELD_COLS].values
        forecast_curve = [model_series(all_fc, BEST_MODEL[c], c).loc[snap_date] for c in YIELD_COLS]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=mats, y=actual_curve, mode="lines+markers",
                                   name="Realized curve", line=dict(color="black", width=3)))
        fig2.add_trace(go.Scatter(x=mats, y=forecast_curve, mode="lines+markers",
                                   name="Forecasted curve (best model per maturity)",
                                   line=dict(color="crimson", dash="dash")))
        fig2.update_layout(height=480, xaxis_title="Maturity (years)", yaxis_title="Yield (%)",
                            xaxis_type="log", title=f"Yield curve — {snap_date.strftime('%Y-%m')}")
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Best model per maturity: " +
            ", ".join(f"{c} → {BEST_MODEL[c]}" for c in YIELD_COLS)
        )

# ========================================================================
# TAB 2: Return / Pricing Engine
# ========================================================================
with tab2:
    st.subheader("Zero-Coupon Bond Return Decomposition")

    engine_model = st.selectbox(
        "Forecast source for expected returns",
        ["Best model per maturity (mixed)"] + ALL_MODELS,
    )

    actual_test = actual.loc[test_dates, YIELD_COLS]

    if engine_model == "Best model per maturity (mixed)":
        # build a forecast curve frame using each column's own best model
        forecast_curves = pd.DataFrame(
            {c: model_series(all_fc, BEST_MODEL[c], c) for c in YIELD_COLS},
            index=test_dates,
        )
    else:
        forecast_curves = pd.DataFrame(
            {c: model_series(all_fc, engine_model, c) for c in YIELD_COLS},
            index=test_dates,
        )

    with st.spinner("Running full-repricing return engine..."):
        results = run_return_engine(actual_test, forecast_curves, target_maturities=MATURITIES)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average monthly return decomposition (realized)**")
        realized = results[results.kind == "realized"]
        comp = realized.groupby("maturity")[["price_return", "roll_return", "carry_return"]].mean()
        comp.index = [f"{t}y" for t in comp.index]

        fig3 = go.Figure()
        for col, color in zip(["price_return", "roll_return", "carry_return"],
                               ["steelblue", "seagreen", "darkorange"]):
            fig3.add_trace(go.Bar(x=comp.index, y=comp[col], name=col, marker_color=color))
        fig3.update_layout(barmode="relative", height=420, yaxis_title="return",
                            title="Price vs. Roll vs. Carry, by maturity")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown("**Directional accuracy: did the forecast get the sign right?**")
        acc = directional_accuracy(results)
        acc.index = [f"{t}y" for t in acc.index]

        fig4 = go.Figure(go.Bar(x=acc.index, y=acc["directional_accuracy"],
                                 marker_color="crimson"))
        fig4.add_hline(y=0.5, line_dash="dash", line_color="grey",
                        annotation_text="coin flip")
        fig4.update_layout(height=420, yaxis_range=[0, 1], yaxis_title="P(correct sign)",
                            title="Directional accuracy by maturity")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**Return engine: raw output**")
    st.dataframe(
        results[results.kind == "realized"]
        .set_index(["date", "maturity"])[["total_return", "price_return", "roll_return", "carry_return"]]
        .round(5)
    )