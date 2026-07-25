from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alphagrid.agents.ingestion_agent import ingest_rss_feed
from alphagrid.agents.orchestrator import process_anomaly_event
from alphagrid.config import load_config
from alphagrid.data.feature_store import build_features, generate_synthetic
from alphagrid.forecasting.anomalies import detect_anomalies
from alphagrid.forecasting.predict import predict_next_hours
from alphagrid.forecasting.train import train_model

# Streamlit Page Config
st.set_page_config(
    page_title="AlphaGrid AI — Grid Forecasting & Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load System Configuration
cfg = load_config()
grid_zone = cfg.get("grid_zone", "DE_LU")

# App Header
st.title("⚡ AlphaGrid AI — Autonomous Time-Series & Market-Intel Platform")
st.caption(
    "Real-time grid forecasting, anomaly detection, and guardrailed LLM thesis synthesis for "
    f"bidding zone **{grid_zone}**."
)

# Sidebar Controls
st.sidebar.header("🎛️ Pipeline Controls")

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
ingest_clicked = st.sidebar.button("Ingest RSS News Feeds")

# Format UTC Strings
start_utc = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).isoformat()
end_utc = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc).isoformat()


# Fetch or Generate Data
@st.cache_data(ttl=600)
def load_dataset(source_type: str, start: str, end: str) -> pd.DataFrame:
    if source_type == "Live ENTSO-E + Open-Meteo":
        try:
            return build_features(start, end, use_cache=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to fetch live ENTSO-E data: {e}. Falling back to synthetic.")
            return generate_synthetic(start, end)
    else:
        return generate_synthetic(start, end)


df = load_dataset(data_source, start_utc, end_utc)

# Model Training Action
if retrain_clicked:
    with st.spinner("Training LightGBM model on dataset..."):
        try:
            mae = train_model(df)
            st.sidebar.success(f"Model trained successfully! Validation MAE: {mae:.2f} MW")
        except Exception as err:  # noqa: BLE001
            st.sidebar.error(f"Training failed: {err}")

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

# Detect Anomalies on Historical Data
actual_series = df["wind_mw"]
anomalies = detect_anomalies(actual_series, actual_series, threshold=threshold)

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
st.subheader("📈 Time-Series Forecast & Prediction Interval")

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

# Anomaly Callouts & Agentic Intel Synthesis Section
st.subheader("🤖 Agentic Market-Intel Synthesis & Guardrail Validation")

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
        "Select Flagged Anomaly Day for LLM Synthesis:", anomaly_options
    )

    selected_index = anomaly_options.index(selected_option)
    selected_anomaly = anomalies[selected_index]

    if st.button("Run Market-Intel Synthesis & Guardrail Check", type="primary"):
        with st.spinner("Synthesizing market thesis with Ollama & validating guardrails..."):
            result = process_anomaly_event(selected_anomaly)

            status = result.get("status", "REJECTED")
            thesis_data = result.get("thesis")
            reason = result.get("reason", "")
            source_chunks = result.get("source_chunks", [])

            if status == "APPROVED" and thesis_data:
                st.success("✅ GUARDRAIL PASSED: Thesis verified (Cosine Sim >= 0.85).")

                with st.expander("📄 Validated Market Thesis Payload", expanded=True):
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
            else:
                st.warning(f"⚠️ GUARDRAIL REJECTED: {reason}")
                if thesis_data:
                    with st.expander("Inspected Rejected Candidate Payload"):
                        st.json(thesis_data)

            if source_chunks:
                with st.expander("📚 Retrieved Source News Chunks (ChromaDB)"):
                    for idx_chunk, chunk in enumerate(source_chunks):
                        st.markdown(f"**Chunk {idx_chunk+1}:** {chunk}")
