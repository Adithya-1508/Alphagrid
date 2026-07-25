# ⚡ AlphaGrid AI — Time-Series Grid Forecasting & Agentic Intelligence Platform

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Docker Support](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checker](https://img.shields.io/badge/types-mypy-informational.svg)](https://mypy-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**AlphaGrid AI** is a production-grade time-series forecasting engine and multi-agent market-intelligence system designed for European power grid operators and energy traders across multiple bidding zones (**DE_LU** — Germany/Luxembourg, **FR** — France, **NL** — Netherlands, and **DK_1** — Denmark West).

It combines real-time grid data ingestion (ENTSO-E), meteorological forecasts (Open-Meteo), gradient boosted decision trees with Optuna tuning (LightGBM), vector-similarity search (ChromaDB), multi-agent Bull vs. Bear debate synthesis, and local LLMs (Ollama `gemma4:e2b`) backed by a **mathematical cosine-similarity guardrail ($\ge 0.85$)** to eliminate hallucinations.

---

## 🏗️ Architecture Overview

```text
                           +---------------------------+
                           |   ENTSO-E + Open-Meteo    |
                           |  (Multi-Zone: DE/FR/NL/DK)|
                           +-------------+-------------+
                                         |
                                         v
+-----------------------+   +------------+------------+   +------------------------+
|   ingestion_agent.py  |   |   LightGBM Forecaster   |   |   artifacts/raw_cache  |
| (RSS -> Clean HTML    |   |  (Optuna Auto-Tuning &  |   |   (Parquet Slices)     |
|   -> ChromaDB vector) |   |   90% Prediction Band)  +------------+-----------+
+-----------+-----------+   +------------+------------+                |
            |                            |                             |
            v                            v                             v
+-----------+-----------+   +------------+------------+   +------------+-----------+
|    debate_agent.py    |   |  detect_anomalies.py    |   |  Streamlit Dashboard   |
| (Bull vs. Bear LLM)   +-->| (Z-Score Residual Days) +-->|  + Executive PDF Brief |
+-----------+-----------+   +------------+------------+   +------------------------+
            |                            
            v                            
+-----------+-----------+                
|      guardrail.py     |                
| (Pydantic Schema &    |                
|  Cosine Sim >= 0.85)  |                
+-----------------------+                
```

---

## ✨ Key Capabilities

1. **Multi-Bidding Zone Ingestion Engine (`src/alphagrid/data/`)**:
   - Ingests actual wind generation from **ENTSO-E REST Platform** and hourly weather metrics from **Open-Meteo** across `DE_LU`, `FR`, `NL`, and `DK_1`.
   - Enforces explicit UTC localization across all pandas `DatetimeIndex` structures.
   - Caches parquet slices locally in `artifacts/raw_cache/` to minimize API latency. Includes a deterministic synthetic fallback generator.

2. **Probabilistic Forecasting & Optuna Tuning Engine (`src/alphagrid/forecasting/`)**:
   - Fits LightGBM regressors with automated **Optuna hyperparameter optimization** (`tune.py`) evaluating cross-validation MAE.
   - Generates 24-to-48 hour ahead forecasts with **90% confidence prediction intervals**.
   - Computes daily residual z-scores to flag grid anomaly days ("Surplus" vs "Shortage").

3. **Multi-Agent Debate & Guardrail Layer (`src/alphagrid/agents/`)**:
   - **Multi-Agent Debate (`debate_agent.py`)**: Runs parallel synthesis between a **Bullish Agent** (upside price catalysts) and **Bearish Agent** (downside oversupply catalysts).
   - **Ingestion Agent**: Strips raw HTML tags from RSS news feeds before embedding text into persistent **ChromaDB** (`artifacts/chroma/`).
   - **Mathematical Guardrail**: Validates candidates via Pydantic (`MarketThesis`), verifies verbatim citation substrings, and enforces embedding cosine similarity $\ge 0.85$ using `sentence-transformers` (`all-MiniLM-L6-v2`).

4. **Interactive Dashboard & Executive PDF Brief (`src/alphagrid/dashboard/`)**:
   - Built with Streamlit, Plotly, and `fpdf2`.
   - Supports multi-zone switching, side-by-side Bull vs. Bear debate card inspection, Optuna hyperparameter study execution, and **one-click Executive PDF Report downloading**.

5. **Portable Docker Containerization (`Dockerfile`, `docker-compose.yml`)**:
   - Fully containerized stack bundling the AlphaGrid Streamlit dashboard and Ollama LLM service for one-command deployment (`docker compose up`).

---

## 📁 Repository Structure

```text
alphagrid/
├── Dockerfile                 # Multi-stage Python 3.12 container image
├── docker-compose.yml         # Container orchestration (App + Ollama service)
├── .env                       # Local secrets (ENTSOE_TOKEN - GitIgnored)
├── .streamlit/
│   └── config.toml            # Streamlit server & file-watcher configuration
├── config/
│   └── config.yaml            # Grid zones map, LLM settings, guardrail threshold
├── artifacts/                 # Local models, parquet raw cache, & ChromaDB
├── src/
│   └── alphagrid/
│       ├── config.py          # Config & .env loader
│       ├── data/              # ENTSO-E, Open-Meteo, & Feature Store
│       │   ├── entsoe_client.py
│       │   ├── weather_client.py
│       │   ├── time_utils.py
│       │   └── feature_store.py
│       ├── forecasting/       # LightGBM, Predictor, Anomaly Detector, & Optuna Tuner
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── anomalies.py
│       │   └── tune.py
│       ├── llm/               # Ollama Client Wrapper
│       │   └── ollama_client.py
│       ├── agents/            # RSS Ingestion, Synthesis, Debate, Guardrail, Orchestrator
│       │   ├── ingestion_agent.py
│       │   ├── guardrail.py
│       │   ├── synthesis_agent.py
│       │   ├── debate_agent.py
│       │   └── orchestrator.py
│       └── dashboard/         # Streamlit App & PDF Brief Generator
│           ├── app.py
│           └── pdf_report.py
├── tests/                     # Automated Test Suite (22/22 passing)
│   ├── test_data.py
│   ├── test_predict.py
│   ├── test_anomalies.py
│   ├── test_agents.py
│   ├── test_dashboard.py
│   └── test_enhancements.py
└── pyproject.toml             # Dependencies & Ruff/Mypy tool configs
```

---

## 🚀 Quick Start Guide

### Option A: Local Python Environment

```powershell
# 1. Install dependencies using uv
uv sync

# 2. Add your ENTSO-E Token to alphagrid/.env
ENTSOE_TOKEN=your_entsoe_api_security_token_here

# 3. Launch Streamlit Dashboard
streamlit run src/alphagrid/dashboard/app.py
```

### Option B: Docker Container Deployment

```powershell
# Run entire stack (App + Ollama service) with Docker Compose
docker compose up --build
```

Access the dashboard at **`http://localhost:8501`**.

---

## 🧪 Testing & Code Quality

Run the automated test suite and static type checkers:

```powershell
# 1. Run full unit test suite (22 tests)
pytest

# 2. Run static type checker (0 errors across 29 files)
mypy src tests

# 3. Run linter and formatting checks
ruff check src tests
ruff format --check .
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
