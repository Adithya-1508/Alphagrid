from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alphagrid.agents.ingestion_agent import ingest_rss_feed
from alphagrid.agents.orchestrator import process_anomaly_event
from alphagrid.config import load_config
from alphagrid.dashboard.pdf_report import generate_executive_pdf
from alphagrid.data.feature_store import build_features, generate_synthetic
from alphagrid.forecasting.anomalies import detect_anomalies
from alphagrid.forecasting.predict import predict_next_hours
from alphagrid.forecasting.train import train_model
from alphagrid.forecasting.tune import tune_hyperparameters

# Streamlit Page Config
st.set_page_config(
    page_title="AlphaGrid AI — Grid Forecasting & Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load System Configuration
cfg = load_config()
grid_zones_map = cfg.get(
    "grid_zones",
    {
        "DE_LU": {"name": "Germany - Luxembourg"},
        "FR": {"name": "France"},
        "NL": {"name": "Netherlands"},
        "DK_1": {"name": "Denmark West (DK1)"},
    },
)

# App Header
st.title("⚡ AlphaGrid AI — Autonomous Time-Series & Market-Intel Platform")
st.caption(
    "Real-time grid forecasting, multi-agent debate synthesis, and guardrailed trading "
    "intelligence."
)

# Sidebar Controls
st.sidebar.header("🎛️ Pipeline Controls")

selected_zone_code = st.sidebar.selectbox(
    "Bidding Zone",
    options=list(grid_zones_map.keys()),
    format_func=lambda x: f"{x} ({grid_zones_map[x].get('name', x)})",
)

data_source = st.sidebar.radio(
    "Data Source",
    options=["Synthetic Fallback", "Live ENTSO-E + Open-Meteo"],
    index=1 if os.getenv("ENTSOE_TOKEN") else 0,
)

# Date Selection
default_end = date.today()
default_start = default_end - timedelta(days=14)
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", default_end)

threshold = st.sidebar.slider(
    "Anomaly Z-Score Threshold", min_value=1.0, max_value=3.0, value=1.5, step=0.1
)

# Actions in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Actions")
retrain_clicked = st.sidebar.button("Re-train LightGBM Model")
tune_clicked = st.sidebar.button("Optuna Auto-Tune Hyperparams")
ingest_clicked = st.sidebar.button("Ingest RSS News Feeds")

# Format UTC Strings
start_utc = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).isoformat()
end_utc = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc).isoformat()


# Fetch or Generate Data
@st.cache_data(ttl=600)
def load_dataset(source_type: str, zone: str, start: str, end: str) -> pd.DataFrame:
    if source_type == "Live ENTSO-E + Open-Meteo":
        try:
            return build_features(start, end, zone=zone, use_cache=True)
        except Exception as e:  # noqa: BLE001
            st.error(
                f"Failed to fetch live ENTSO-E data for {zone}: {e}. Falling back to synthetic."
            )
            return generate_synthetic(start, end)
    else:
        return generate_synthetic(start, end)


df = load_dataset(data_source, selected_zone_code, start_utc, end_utc)

# Model Training Action
if retrain_clicked:
    with st.spinner(f"Training LightGBM model on {selected_zone_code} dataset..."):
        try:
            mae = train_model(df)
            st.sidebar.success(f"Model trained! Validation MAE: {mae:.2f} MW")
        except Exception as err:  # noqa: BLE001
            st.sidebar.error(f"Training failed: {err}")

# Optuna Hyperparameter Tuning Action
if tune_clicked:
    with st.spinner("Running Optuna Automated Hyperparameter Study (15 trials)..."):
        try:
            best_params, best_mae = tune_hyperparameters(df, n_trials=15)
            st.sidebar.success(f"Optuna Study Complete! Best CV MAE: {best_mae:.2f} MW")
            st.sidebar.json(best_params)
        except Exception as err:  # noqa: BLE001
            st.sidebar.error(f"Optuna tuning failed: {err}")

# RSS Ingestion Action
if ingest_clicked:
    with st.spinner("Ingesting market news feeds into ChromaDB..."):
        try:
            feed_url = "https://feeds.bbci.co.uk/news/world/rss.xml"
            count = ingest_rss_feed(feed_url)
            st.sidebar.success(f"Ingested {count} news items into ChromaDB vector store.")
        except Exception as err:  # noqa: BLE001
            st.sidebar.error(f"Ingestion failed: {err}")

# Generate Predictions & Detect Anomalies
try:
    forecast_df = predict_next_hours(df, horizon_hours=24)
except Exception:  # noqa: BLE001
    # Auto-train if model missing
    train_model(df)
    forecast_df = predict_next_hours(df, horizon_hours=24)

# Detect Anomalies on Historical Data vs Rolling Baseline
actual_series = df["wind_mw"]
baseline_forecast = df["wind_mw"].rolling(24, min_periods=1).mean()
anomalies = detect_anomalies(actual_series, baseline_forecast, threshold=threshold)

# Metrics Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Data Points", f"{len(df)} hrs")
with col2:
    avg_gen = df["wind_mw"].mean()
    st.metric("Average Wind Gen", f"{avg_gen:,.0f} MW")
with col3:
    max_gen = df["wind_mw"].max()
    st.metric("Peak Generation", f"{max_gen:,.0f} MW")
with col4:
    st.metric("Flagged Anomaly Days", f"{len(anomalies)} days")

st.markdown("---")

# Main Visualization Section
st.subheader(f"📈 Time-Series Forecast & Prediction Interval — {selected_zone_code}")

# Construct Plotly Figure
fig = go.Figure()

# Historical Actuals
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["wind_mw"],
        mode="lines",
        name="Actual Wind MW",
        line=dict(color="#1f77b4", width=2),
    )
)

# 24h Point Forecast
fig.add_trace(
    go.Scatter(
        x=forecast_df.index,
        y=forecast_df["forecast"],
        mode="lines+markers",
        name="24h Forecast",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    )
)

# Prediction Interval Band (Upper & Lower)
fig.add_trace(
    go.Scatter(
        x=forecast_df.index,
        y=forecast_df["upper_bound"],
        mode="lines",
        name="Upper Bound (90%)",
        line=dict(width=0),
        showlegend=False,
    )
)

fig.add_trace(
    go.Scatter(
        x=forecast_df.index,
        y=forecast_df["lower_bound"],
        mode="lines",
        name="90% Prediction Interval",
        fill="tonexty",
        fillcolor="rgba(255, 127, 14, 0.2)",
        line=dict(width=0),
    )
)

# Plot Anomaly Points if present
if anomalies:
    assert isinstance(df.index, pd.DatetimeIndex)
    anomaly_dates = [a["date"] for a in anomalies]
    daily_df = df.groupby(df.index.date)["wind_mw"].mean()
    anomaly_vals = [daily_df.get(pd.to_datetime(d).date(), avg_gen) for d in anomaly_dates]

    fig.add_trace(
        go.Scatter(
            x=[pd.to_datetime(d) for d in anomaly_dates],
            y=anomaly_vals,
            mode="markers",
            name="Grid Anomaly Days",
            marker=dict(size=12, color="red", symbol="x"),
        )
    )

fig.update_layout(
    height=500,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Timestamp (UTC)",
    yaxis_title="Generation (MW)",
    hovermode="x unified",
    template="plotly_white",
)

st.plotly_chart(fig)

st.markdown("---")

# Multi-Agent Debate & Agentic Intel Synthesis Section
st.subheader("🤖 Multi-Agent Bull vs. Bear Debate & Guardrail Validation")

if not anomalies:
    st.info(
        "No anomaly days flagged at the current z-score threshold. "
        "Adjust the slider in the sidebar to inspect milder deviations."
    )
else:
    anomaly_options = [
        f"{a['date']} — {a['direction']} (Mag: {a['magnitude']:.0f} MW, Z: {a['zscore']:.2f})"
        for a in anomalies
    ]
    selected_option = st.selectbox(
        "Select Flagged Anomaly Day for Multi-Agent Debate:", anomaly_options
    )

    selected_index = anomaly_options.index(selected_option)
    selected_anomaly = anomalies[selected_index]
    selected_anomaly["market_symbol"] = selected_zone_code

    if st.button("Execute Multi-Agent Debate & Guardrail Check", type="primary"):
        with st.spinner("Running Bull vs. Bear Agent Debate & evaluating Guardrail..."):
            result = process_anomaly_event(selected_anomaly, use_debate=True)

            status = result.get("status", "REJECTED")
            thesis_data = result.get("thesis")
            bull_data = result.get("bull_thesis")
            bear_data = result.get("bear_thesis")
            reason = result.get("reason", "")
            source_chunks = result.get("source_chunks", [])

            # Display Bull vs Bear Debate Cards side-by-side
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("### 🐂 Bullish Agent Candidate")
                if bull_data:
                    b_pass = "✅ PASS" if result.get("bull_valid") else "❌ FAIL"
                    st.caption(f"Guardrail Check: {b_pass}")
                    st.write(bull_data.get("reasoning", ""))
                else:
                    st.write("No Bull thesis generated.")

            with d_col2:
                st.markdown("### 🐻 Bearish Agent Candidate")
                if bear_data:
                    br_pass = "✅ PASS" if result.get("bear_valid") else "❌ FAIL"
                    st.caption(f"Guardrail Check: {br_pass}")
                    st.write(bear_data.get("reasoning", ""))
                else:
                    st.write("No Bear thesis generated.")

            st.markdown("---")

            if status == "APPROVED" and thesis_data:
                st.success("✅ GUARDRAIL APPROVED: Winning thesis verified (Cosine Sim >= 0.85).")

                with st.expander("📄 Validated Winning Market Thesis Payload", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**Market Symbol:** `{thesis_data.get('market_symbol')}`")
                    with c2:
                        pos = thesis_data.get("position_direction")
                        direction_color = "🟢" if pos == "Long" else "🔴"
                        st.markdown(f"**Position:** {direction_color} `{pos}`")
                    with c3:
                        horizon = thesis_data.get("target_horizon_hours")
                        st.markdown(f"**Target Horizon:** `{horizon}h`")

                    st.markdown("### Reasoning")
                    st.write(thesis_data.get("reasoning", ""))

                    st.markdown("### Verbatim Citations")
                    for cit in thesis_data.get("verbatim_citations", []):
                        st.info(f'"{cit}"')

                # PDF Download Button
                pdf_bytes = generate_executive_pdf(selected_anomaly, thesis_data, source_chunks)
                st.download_button(
                    label="📄 Download Executive PDF Brief",
                    data=pdf_bytes,
                    file_name=f"alphagrid_intel_brief_{selected_zone_code}_{selected_anomaly.get('date')}.pdf",
                    mime="application/pdf",
                )

            else:
                st.warning(f"⚠️ GUARDRAIL REJECTED: {reason}")
                if thesis_data:
                    with st.expander("Inspected Rejected Candidate Payload"):
                        st.json(thesis_data)

            if source_chunks:
                with st.expander("📚 Retrieved Source News Chunks (ChromaDB)"):
                    for idx_chunk, chunk in enumerate(source_chunks):
                        st.markdown(f"**Chunk {idx_chunk + 1}:** {chunk}")
