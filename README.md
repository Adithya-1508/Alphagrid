# ⚡ AlphaGrid AI — Time-Series Grid Forecasting & Agentic Intelligence Platform

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checker](https://img.shields.io/badge/types-mypy-informational.svg)](https://mypy-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**AlphaGrid AI** is a production-grade time-series forecasting engine and agentic market-intelligence system designed for power grid operators and energy traders (specifically bidding zone **DE_LU** — Germany–Luxembourg).

It combines high-frequency grid data ingestion (ENTSO-E), meteorological forecasts (Open-Meteo), gradient boosted decision trees (LightGBM), vector-similarity search (ChromaDB), and local LLMs (Ollama `gemma4:e2b`) backed by a **mathematical cosine-similarity guardrail ($\ge 0.85$)** to eliminate hallucinations.

---

## 🏗️ Architecture Overview

```text
                           +---------------------------+
                           |   ENTSO-E + Open-Meteo    |
                           |  (Strict UTC Localized)   |
                           +-------------+-------------+
                                         |
                                         v
+-----------------------+   +------------+------------+   +------------------------+
|  ingestion_agent.py   |   |   LightGBM Forecaster   |   |   artifacts/raw_cache  |
| (RSS -> Clean HTML    |   |  (TimeSeriesSplit 5-fold+ |   |   (Parquet Slices)     |
|   -> ChromaDB vector) |   |   90% Prediction Interval)  +------------+-----------+
+-----------+-----------+   +------------+------------+                |
            |                            |                             |
            v                            v                             v
+-----------+-----------+   +------------+------------+   +------------+-----------+
|  synthesis_agent.py   |   |  detect_anomalies.py    |   |  Streamlit Dashboard   |
| (Ollama Gemma 4 LLM)  +-->| (Z-Score Residual Days) +-->|  (Plotly Interactive)  |
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

1. **Strict UTC Ingestion Engine (`src/alphagrid/data/`)**:
   - Ingests actual wind generation from **ENTSO-E REST Platform** and hourly wind speed / temperature from **Open-Meteo**.
   - Enforces explicit UTC localization across all pandas `DatetimeIndex` structures to eliminate daylight saving time (DST) boundary bugs.
   - Caches parquet slices locally in `artifacts/raw_cache/` to minimize API latency and respect rate limits. Includes a deterministic synthetic fallback generator for offline testing.

2. **Probabilistic Forecasting & Anomaly Engine (`src/alphagrid/forecasting/`)**:
   - Fits LightGBM regressors using out-of-sample temporal feature engineering (lags, rolling stats, hour/month cyclical encodings).
   - Generates 24-to-48 hour ahead forecasts with **90% confidence prediction intervals** (clipped lower bound at $0\text{ MW}$).
   - Computes daily residual z-scores to flag grid anomaly days ("Surplus" vs "Shortage").

3. **Agentic Market-Intel Layer (`src/alphagrid/agents/`)**:
   - **Ingestion Agent**: Strips raw HTML tags (`<p>`, `<a>`, `<br>`) from RSS news feeds before embedding text into persistent **ChromaDB** (`artifacts/chroma/`).
   - **Synthesis Agent**: Prompts local Ollama (`gemma4:e2b`, `temperature=0`) to construct structured trading theses for flagged anomaly days.
   - **Mathematical Guardrail**: Validates candidate theses via Pydantic (`MarketThesis`), verifies verbatim citation substrings, and enforces embedding cosine similarity $\ge 0.85$ using `sentence-transformers` (`all-MiniLM-L6-v2`). Unvalidated theses are dropped immediately.

4. **Interactive Dashboard (`src/alphagrid/dashboard/app.py`)**:
   - Built with Streamlit and Plotly. Visualizes live actual generation curves, point forecasts, shaded confidence bands, red anomaly marker overlays, and guardrailed Market Thesis recommendation panels.

---

## 📁 Repository Structure

```text
alphagrid/
├── .env                       # Local secrets (ENTSOE_TOKEN - GitIgnored)
├── .streamlit/
│   └── config.toml            # Streamlit server & file-watcher configuration
├── config/
│   └── config.yaml            # Grid zone, model hyperparams, guardrail threshold
├── artifacts/                 # Local models, parquet raw cache, & ChromaDB
├── src/
│   └── alphagrid/
│       ├── config.py          # Config & .env loader
│       ├── data/              # ENTSO-E, Open-Meteo, & Feature Store
│       │   ├── entsoe_client.py
│       │   ├── weather_client.py
│       │   ├── time_utils.py
│       │   └── feature_store.py
│       ├── forecasting/       # LightGBM, Predictor, & Anomaly Detector
│       │   ├── train.py
│       │   ├── predict.py
│       │   └── anomalies.py
│       ├── llm/               # Ollama Client Wrapper
│       │   └── ollama_client.py
│       ├── agents/            # RSS Ingestion, Synthesis, Guardrail, Orchestrator
│       │   ├── ingestion_agent.py
│       │   ├── guardrail.py
│       │   ├── synthesis_agent.py
│       │   └── orchestrator.py
│       └── dashboard/         # Streamlit App
│           └── app.py
├── tests/                     # Automated Test Suite (19/19 passing)
│   ├── test_data.py
│   ├── test_predict.py
│   ├── test_anomalies.py
│   ├── test_agents.py
│   └── test_dashboard.py
└── pyproject.toml             # Dependencies & Ruff/Mypy tool configs
```

---

## 🚀 Quick Start Guide

### 1. Installation

Clone the repository and install dependencies using `uv` (recommended) or `pip`:

```powershell
# Using uv (fastest)
uv sync

# Or using standard pip virtualenv
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

### 2. Environment Configuration

Create a `.env` file in `alphagrid/.env`:

```env
ENTSOE_TOKEN=your_entsoe_api_security_token_here
```

### 3. Launching the Streamlit Dashboard

Run the interactive dashboard UI:

```powershell
streamlit run src/alphagrid/dashboard/app.py
```

Access the UI in your web browser at **`http://localhost:8501`**.

---

## 🧪 Testing & Code Quality

Run the automated test suite and static type checkers:

```powershell
# 1. Run full unit test suite (19 tests)
pytest

# 2. Run static type checker
mypy src tests

# 3. Run linter and formatting checks
ruff check src tests
ruff format --check .
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
