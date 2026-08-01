from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alphagrid.agents.graph_rag import query_graph_context
from alphagrid.agents.ingestion_agent import ingest_rss_feed
from alphagrid.agents.orchestrator import process_anomaly_event
from alphagrid.config import load_config
from alphagrid.dashboard.pdf_report import generate_executive_pdf
from alphagrid.data.feature_store import build_features, generate_synthetic
from alphagrid.forecasting.anomalies import detect_anomalies
from alphagrid.forecasting.explainability import compute_feature_explainability
from alphagrid.forecasting.mlops_pipeline import check_and_trigger_retrain
from alphagrid.forecasting.predict import predict_next_hours
from alphagrid.forecasting.scenario_engine import run_monte_carlo_scenarios
from alphagrid.forecasting.train import FEATURES, train_model
from alphagrid.forecasting.tune import tune_hyperparameters
from alphagrid.trading.backtest import run_backtest

# Streamlit Page Config
st.set_page_config(
    page_title="AlphaGrid AI — Enterprise Trading & Intelligence Desk",
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
st.title("⚡ AlphaGrid AI — Institutional Trading Desk & MLOps Command Center")
st.caption(
    "Real-time grid forecasting, multi-agent debate synthesis, GraphRAG, "
    "Monte Carlo stress testing, and automated MLOps continuous retraining."
)

# Sidebar Controls
st.sidebar.header("🎛️ Pipeline Controls")

selected_zone_code = str(
    st.sidebar.selectbox(
        "Bidding Zone",
        options=list(grid_zones_map.keys()),
        format_func=lambda x: f"{x} ({grid_zones_map[x].get('name', x)})",
    )
    or "DE_LU"
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
st.sidebar.subheader("🔄 MLOps & Data Actions")
retrain_clicked = st.sidebar.button("Re-train ML Models")
mlops_retrain_clicked = st.sidebar.button("Run MLOps Drift Retrain Check")
tune_clicked = st.sidebar.button("Optuna Auto-Tune Hyperparams")
ingest_clicked = st.sidebar.button("Ingest RSS News Feeds")

# Format UTC Strings
start_utc = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc).isoformat()
end_utc = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc).isoformat()


# Fetch or Generate Data
@st.cache_data(ttl=600)
def load_dataset(source_type: str, zone: str | None, start: str, end: str) -> pd.DataFrame:
    effective_zone = zone or "DE_LU"
    if source_type == "Live ENTSO-E + Open-Meteo":
        try:
            return build_features(start, end, zone=effective_zone, use_cache=True)
        except Exception as e:  # noqa: BLE001
            st.error(
                f"Failed to fetch live ENTSO-E data for {effective_zone}: {e}. "
                "Falling back to synthetic data."
            )
            return generate_synthetic(start, end)
    else:
        return generate_synthetic(start, end)


df = load_dataset(data_source, selected_zone_code, start_utc, end_utc)

# Model Training Action
if retrain_clicked:
    with st.spinner(f"Training ML models on {selected_zone_code} dataset..."):
        try:
            mae = train_model(df)
            st.sidebar.success(f"Models trained! Validation MAE: {mae:.2f} MW")
        except Exception as err:  # noqa: BLE001
            st.sidebar.error(f"Training failed: {err}")

# MLOps Drift Check & Retrain Action
if mlops_retrain_clicked:
    with st.spinner("Evaluating statistical data drift & triggering MLOps retraining..."):
        try:
            ref_data = generate_synthetic(start_utc, end_utc)
            mlops_status = check_and_trigger_retrain(ref_data, df, force_retrain=True)
            st.sidebar.success(f"MLOps Retrain Complete! Status: {mlops_status.status_message}")
        except Exception as err:  # noqa: BLE001
            st.sidebar.error(f"MLOps pipeline failed: {err}")

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
except Exception as exc:  # noqa: BLE001
    st.warning(f"Model prediction fallback triggered: {exc}")
    train_model(df)
    forecast_df = predict_next_hours(df, horizon_hours=24)

# Detect Anomalies on Historical Data vs Rolling Baseline (Shifted to eliminate lookahead bias)
actual_series = df["wind_mw"]
baseline_forecast = df["wind_mw"].shift(1).rolling(24, min_periods=1).mean().bfill()
anomalies = detect_anomalies(actual_series, baseline_forecast, threshold=threshold)

# Backtest Trading Strategy Metrics
backtest_res = run_backtest(df, forecast_df, initial_capital_eur=100_000.0)
b_metrics = backtest_res.metrics

# Metrics Bar
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Strategy PnL (€)", f"€{b_metrics.total_pnl_eur:,.2f}")
with col2:
    st.metric("Sharpe Ratio", f"{b_metrics.sharpe_ratio:.2f}")
with col3:
    st.metric("Max Drawdown", f"{b_metrics.max_drawdown_pct:.2f}%")
with col4:
    st.metric("95% VaR (€)", f"€{b_metrics.var_95_eur:,.2f}")
with col5:
    st.metric("Win Rate (%)", f"{b_metrics.win_rate_pct:.1f}%")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Time-Series Forecasts",
        "🌪️ Monte Carlo Stress Testing",
        "🕸️ GraphRAG Knowledge Base",
        "🤖 Multi-Agent Debate Desk",
    ]
)

with tab1:
    st.subheader(f"📈 Probabilistic Forecasts & Feature Attribution — {selected_zone_code}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["wind_mw"],
            mode="lines",
            name="Actual Wind MW",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["forecast"],
            mode="lines+markers",
            name="24h Forecast (P50)",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["upper_bound"],
            mode="lines",
            name="P90 Bound",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["lower_bound"],
            mode="lines",
            name="P10-P90 Interval",
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.2)",
            line=dict(width=0),
        )
    )

    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Timestamp (UTC)",
        yaxis_title="Generation (MW)",
        hovermode="x unified",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature Attribution Breakdown
    st.markdown("### 🔍 Model Feature Importance & Attribution Breakdown")
    active_feats = [f for f in FEATURES if f in df.columns]
    explainability = compute_feature_explainability(None, active_feats)

    f_cols = st.columns(len(explainability.top_features[:4]))
    for idx, feat in enumerate(explainability.top_features[:4]):
        with f_cols[idx]:
            st.metric(feat.feature_name, f"{feat.contribution_pct:.1f}%")

with tab2:
    st.subheader("🌪️ Monte Carlo Grid Stress Testing & Scenario Engine")
    scenario_type = st.selectbox("Select Stress Scenario:", ["Dunkelflaute", "Heatwave", "Outage"])
    sim_count = st.slider("Number of Monte Carlo Paths:", 20, 200, 50, 10)

    if st.button("Run Monte Carlo Stress Test", type="primary"):
        with st.spinner("Simulating stochastic grid stress paths..."):
            s_res = run_monte_carlo_scenarios(df, scenario=scenario_type, num_simulations=sim_count)
            s_m = s_res.metrics

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Risk Status", s_m.risk_level)
            with sc2:
                st.metric("Mean Forecast", f"{s_m.mean_forecast_mw:,.0f} MW")
            with sc3:
                st.metric("5th Percentile Deficit (P05)", f"{s_m.p05_deficit_mw:,.0f} MW")
            with sc4:
                st.metric("Max Price Spike", f"€{s_m.max_price_spike_eur_mwh:.2f}/MWh")

            mc_fig = go.Figure()
            for p_idx, path in enumerate(s_res.simulated_paths):
                mc_fig.add_trace(
                    go.Scatter(
                        y=path,
                        mode="lines",
                        name=f"Path {p_idx + 1}",
                        opacity=0.6,
                    )
                )
            mc_fig.update_layout(
                title=f"Monte Carlo Simulated Paths under {scenario_type} Shock",
                xaxis_title="Forecast Hours",
                yaxis_title="Generation (MW)",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(mc_fig, use_container_width=True)

with tab3:
    st.subheader(f"🕸️ GraphRAG Knowledge Base & Traversal Context — {selected_zone_code}")
    graph_ctx = query_graph_context(selected_zone_code)
    st.info(graph_ctx)

with tab4:
    st.subheader("🤖 Multi-Agent Bull vs. Bear Debate & Guardrail Validation")
    if not anomalies:
        st.info("No anomaly days flagged at current z-score threshold.")
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
                    st.success(
                        "✅ GUARDRAIL APPROVED: Winning thesis verified (Cosine Sim >= 0.85)."
                    )

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
