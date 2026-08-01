# ⚡ AlphaGrid AI - Enterprise Time-Series Grid Forecasting & Agentic Intelligence Platform

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Docker Support](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checker](https://img.shields.io/badge/types-mypy-informational.svg)](https://mypy-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**AlphaGrid AI** is an end-to-end European grid forecasting pipeline ingesting real-time ENTSO-E telemetry. Combines LightGBM time-series models with multi-agent RAG guardrails (ChromaDB + LangChain) for real-time anomaly contextualization.

Engineered for European power grid operators, quantitative analysts, and energy trading desks across key ENTSO-E bidding zones:
* **`DE_LU`** — Germany / Luxembourg
* **`FR`** — France
* **`NL`** — Netherlands
* **`DK_1`** — Denmark West

The platform bridges real-time physical power telemetry with multi-agent natural language intelligence to deliver actionable market signals. It combines real-time grid ingestion (**ENTSO-E REST API**), high-resolution meteorological data (**Open-Meteo API**), gradient boosted decision trees with hyperparameter search (**LightGBM + Optuna**), vector-similarity search (**ChromaDB**), parallel **Bull vs. Bear debate synthesis**, and local LLMs (**Ollama** `gemma4:e2b`) backed by a **mathematical cosine-similarity guardrail** to eliminate model hallucinations.

---

## 🎯 Trading Edge & Commercial Use Cases

1. **Renewable Imbalance & Dunkelflaute Forecasting:** Detects sudden wind generation drop-offs across Central European bidding zones, enabling intraday traders to position against physical grid shortfalls before imbalance settlement prices spike.
2. **Inter-Zone Flow & Congestion Arbitrage:** Evaluates cross-border HVDC/AC transmission limits and regional supply-demand divergence between neighboring zones (e.g., `DE_LU` ↔ `FR`).
3. **Automated REMIT Market Intelligence:** Ingests unstructured energy market news and REMIT Urgent Market Messages (UMMs), extracting asset trip capacities and parsing market sentiment without manual review.
4. **Deterministic Tail-Risk Guardrails:** Ensures automated market summaries cite exact source substrings and pass strict semantic embedding thresholds before reaching trading desk executives.
5. **Monte Carlo Stochastic Stress Testing:** Simulates 1,000+ extreme weather and outage paths under Dunkelflaute, Heatwave, and Baseload trip scenarios to calculate Value-at-Risk ($95\%$ VaR) and spot price spikes.

---

## 🏗️ Complete 14-Level Architecture Overview

```text
                            +---------------------------+
                            |   ENTSO-E + Open-Meteo    |
                            |  (Multi-Zone: DE/FR/NL/DK)|
                            +-------------+-------------+
                                          |
                                          v
+-----------------------+   +------------+------------+   +------------------------+
|  ingestion_agent.py   |   |   LightGBM + Sequence   |   |   artifacts/raw_cache  |
| (RSS -> Clean HTML    |   | (Quantile Regressors &  |   |   (Parquet & Feast)    |
|   -> ChromaDB vector) |   |  Conformal Predictor)   +------------+-----------+
+-----------+-----------+   +------------+------------+                |
            |                            |                             |
            v                            v                             v
+-----------+-----------+   +------------+------------+   +------------+-----------+
|    debate_agent.py    |   |  detect_anomalies.py    |   |   Streamlit Trading    |
| (Bull vs. Bear LLM)   +-->| (Z-Score Residual Days) +-->|      Command Center    |
+-----------+-----------+   +------------+------------+   +------------------------+
            |                            |                             |
            v                            v                             v
+-----------+-----------+   +------------+------------+   +------------+-----------+
|     guardrail.py      |   |   scenario_engine.py    |   |   mlops_pipeline.py    |
| (Pydantic Schema &    |   | (Monte Carlo Stress Test|   | (Automated PSI Drift   |
|  Cosine Sim >= 0.85)  |   | & Value-at-Risk 95% VaR)|   |  & Continuous Retrain) |
+-----------------------+   +-------------------------+   +------------------------+
```

---

## 🔬 Mathematical Formulations & Methodology

### 1. Probabilistic Prediction Interval & Conformal Calibration
LightGBM fits quantile loss functions at $\alpha/2 = 0.05$ and $1 - \alpha/2 = 0.95$ to output a **90% confidence prediction band**, calibrated using Split Conformal Prediction score quantiles:

$$\text{Confidence Band}_t = \left[ \hat{y}_{t, \, 0.05} - q_{1-\alpha}, \, \hat{y}_{t, \, 0.95} + q_{1-\alpha} \right]$$

### 2. Residual Anomaly Z-Score
Systemic imbalance days ("Surplus" vs "Shortage") are detected by calculating normalized residual deviations $e_t = y_t - \hat{y}_t$ over rolling time windows:

$$Z_t = \frac{e_t - \mu_e}{\sigma_e}$$

Anomalies are flagged when $|Z_t| \ge 2.0$, signaling abnormal weather shifts or transmission constraints.

### 3. Statistical Data Drift (PSI & KS-Test)
Dataset drift between baseline reference distributions $P$ and live inference distributions $Q$ is measured across features using Population Stability Index (PSI):

$$\text{PSI} = \sum_{b=1}^{B} \left( P_b - Q_b \right) \times \ln\left(\frac{P_b}{Q_b}\right)$$

When $\text{PSI} > 0.20$ or Kolmogorov-Smirnov $p < 0.05$, the **MLOps Retraining Pipeline** automatically triggers model retraining.

### 4. Embedding Cosine-Similarity Guardrail
LLM-generated market thesis candidates $\mathbf{v}_{\text{llm}}$ are embedded via `sentence-transformers` (`all-MiniLM-L6-v2`) and compared against the grounded RAG context vector $\mathbf{u}_{\text{ctx}}$:

$$S_{\cos}(\mathbf{u}_{\text{ctx}}, \mathbf{v}_{\text{llm}}) = \frac{\mathbf{u}_{\text{ctx}} \cdot \mathbf{v}_{\text{llm}}}{\|\mathbf{u}_{\text{ctx}}\| \|\mathbf{v}_{\text{llm}}\|} \ge 0.85$$

Responses scoring $S_{\cos} < 0.85$ or failing Pydantic schema verification are automatically rejected and re-routed for deterministic fallback synthesis.

---

## ✨ Comprehensive 14-Level Platform Capabilities

1. **Level 1 — Statistical Data Drift Engine (`drift.py`)**: Population Stability Index (PSI) & Kolmogorov-Smirnov distribution drift monitoring.
2. **Level 2 — European Energy Market Features (`market_prices.py`)**: EEX day-ahead power prices, TTF gas, and EU ETS carbon allowance pricing models.
3. **Level 3 — Agent Reflection & SQLite Long-Term Memory (`memory_store.py`)**: Persistent SQLite memory store tracking past trade outcomes and agent self-corrections.
4. **Level 4 — Knowledge Graph RAG (`graph_rag.py`)**: NetworkX entity-relationship graph linking bidding zones, interconnectors, and power stations with ChromaDB vector search.
5. **Level 5 — Backtest Simulation & Algorithmic Trading Desk (`backtest.py`)**: Realistic trade simulation calculating PnL (€), Sharpe Ratio, Max Drawdown %, 95% VaR, and Win Rate.
6. **Level 6 — Model Explainability (`explainability.py`)**: SHAP-inspired feature attribution breaking down model prediction drivers.
7. **Level 7 — Real-Time Event Streaming (`event_stream.py`)**: Async pub/sub event broker handling real-time telemetry updates.
8. **Level 8 — Conformal Prediction (`conformal.py`)**: Calibrated distribution-free non-conformity score bounds guaranteeing coverage.
9. **Level 9 — Deep Learning Sequence Models (`sequence_models.py`)**: Auto-regressive sequence models for multi-horizon generation forecasting.
10. **Level 10 — Automated Feature Store (`feast_store.py`)**: Point-in-time offline feature joins & online low-latency materialization store.
11. **Level 11 — Multi-Grid Interconnector Power Flow Model (`interconnectors.py`)**: Cross-border transmission capacity and congestion spread solver.
12. **Level 12 — Scenario Engine & Monte Carlo Stress Testing (`scenario_engine.py`)**: Stochastic Dunkelflaute, Heatwave, and Outage path simulation.
13. **Level 13 — Continuous Retraining & MLOps Pipeline (`mlops_pipeline.py`)**: Automated drift monitoring and model retraining orchestrator.
14. **Level 14 — Enterprise Risk Dashboard & Institutional Export Center (`app.py`)**: Tabbed Streamlit trading floor application with executive PDF reporting.

---

## 📁 Repository Structure

```text
alphagrid/
├── Dockerfile                  # Multi-stage Python 3.12 container image
├── docker-compose.yml          # Container orchestration (App + Ollama service)
├── .env                        # Local environment secrets (ENTSOE_TOKEN - GitIgnored)
├── .streamlit/
│   └── config.toml             # Streamlit server & file-watcher configuration
├── config/
│   └── config.yaml             # Bidding zones map, LLM params, guardrail thresholds
├── artifacts/                  # Local model binaries, Parquet cache, ChromaDB & SQLite memory
├── src/
│   └── alphagrid/
│       ├── config.py           # Configuration & .env loader
│       ├── data/               # Feature Store, ENTSO-E, Market Prices, Interconnectors
│       │   ├── entsoe_client.py
│       │   ├── weather_client.py
│       │   ├── market_prices.py
│       │   ├── interconnectors.py
│       │   ├── feast_store.py
│       │   ├── time_utils.py
│       │   └── feature_store.py
│       ├── forecasting/        # LightGBM Engine, Sequence Models, Conformal, Drift & MLOps
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── anomalies.py
│       │   ├── drift.py
│       │   ├── ensemble.py
│       │   ├── conformal.py
│       │   ├── sequence_models.py
│       │   ├── scenario_engine.py
│       │   ├── mlops_pipeline.py
│       │   ├── explainability.py
│       │   └── tune.py
│       ├── llm/                # Local Ollama Client Wrapper
│       │   └── ollama_client.py
│       ├── trading/            # Algorithmic Trading & Backtest Simulation
│       │   └── backtest.py
│       ├── agents/             # RSS Ingestion, GraphRAG, Memory Store, Debate & Guardrails
│       │   ├── ingestion_agent.py
│       │   ├── memory_store.py
│       │   ├── reflection_agent.py
│       │   ├── graph_rag.py
│       │   ├── guardrail.py
│       │   ├── synthesis_agent.py
│       │   ├── debate_agent.py
│       │   └── orchestrator.py
│       └── dashboard/          # Streamlit Interactive Desk & Executive PDF Brief Generator
│           ├── app.py
│           └── pdf_report.py
├── tests/                      # Automated Unit & Integration Test Suite (49/49 passing)
└── pyproject.toml              # Project dependencies, Ruff, & Mypy configurations
```

---

## 🚀 Quick Start Guide

### Option A: Local Setup (`uv`)

```powershell
# 1. Install dependencies using uv
uv sync

# 2. Add your ENTSO-E API Security Token to .env (Optional - fallback provided)
ENTSOE_TOKEN=your_entsoe_api_security_token_here

# 3. Launch Streamlit Trading Command Center
streamlit run src/alphagrid/dashboard/app.py
```

Access the interactive trading portal at **`http://localhost:8501`**.

### Option B: Docker Container Deployment
```powershell
# Build and spin up the complete application stack (Dashboard + Ollama)
docker compose up --build
```

---

### 🧪 Code Quality Assurance & Automated Testing

The codebase adheres to strict software engineering standards, fully verified via automated tests and static analysis:

```powershell
# 1. Execute full unit and integration test suite (49/49 passing)
pytest

# 2. Run static type analysis (0 type errors across 60 source files)
mypy src tests

# 3. Run linter and code formatting checks
ruff check src tests
ruff format --check .
```

---

### 📜 License
Distributed under the MIT License. See LICENSE for details.
