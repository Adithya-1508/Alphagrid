# ⚡ AlphaGrid AI — Time-Series Grid Forecasting & Agentic Intelligence Platform

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![Docker Support](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checker](https://img.shields.io/badge/types-mypy-informational.svg)](https://mypy-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**AlphaGrid AI** is a production-grade time-series forecasting engine and multi-agent market-intelligence platform engineered for European power grid operators, quantitative analysts, and energy trading desks across key ENTSO-E bidding zones:
* **`DE_LU`** — Germany / Luxembourg
* **`FR`** — France
* **`NL`** — Netherlands
* **`DK_1`** — Denmark West

The platform bridges real-time physical power telemetry with multi-agent natural language intelligence to deliver actionable market signals. It combines real-time grid ingestion (**ENTSO-E REST API**), high-resolution meteorological data (**Open-Meteo API**), gradient boosted decision trees with hyperparameter search (**LightGBM + Optuna**), vector-similarity search (**ChromaDB**), parallel **Bull vs. Bear debate synthesis**, and local LLMs (**Ollama** `gemma4:e2b`) backed by a **mathematical cosine-similarity guardrail** to eliminate model hallucinations.

---

## 🎯 Trading Edge & Commercial Use Cases

1. **Renewable Imbalance & Dunkelflaute Forecasting:** Detects sudden wind generation drop-offs across Central European bidding zones, enabling intraday traders to position against physical grid shortfalls before imbalance settlement prices spike.
2. **Inter-Zone Flow & Congestion Arbitrage:** Evaluates cross-border transmission limits and regional supply-demand divergence between neighboring zones (e.g., `DE_LU` $\leftrightarrow$ `FR`).
3. **Automated REMIT Market Intelligence:** Ingests unstructured energy market news and REMIT Urgent Market Messages (UMMs), extracting asset trip capacities and parsing market sentiment without manual review.
4. **Deterministic Tail-Risk Guardrails:** Ensures automated market summaries cite exact source substrings and pass strict semantic embedding thresholds before reaching trading desk executives.

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
|    debate_agent.py    |   |  detect_anomalies.py    |   |   Streamlit Dashboard  |
| (Bull vs. Bear LLM)   +-->| (Z-Score Residual Days) +-->|  + Executive PDF Brief |
+-----------+-----------+   +------------+------------+   +------------------------+
            |                                
            v                                
+-----------+-----------+                    
|     guardrail.py      |                    
| (Pydantic Schema &    |                    
|  Cosine Sim >= 0.85)  |                    
+-----------------------+
```
## 🔬 Mathematical Formulations & Methodology

### 1. Probabilistic Prediction Interval
Rather than issuing single point forecasts, LightGBM fits quantile loss functions at $\alpha/2 = 0.05$ and $1 - \alpha/2 = 0.95$ to output a **90% confidence prediction band**:

$$\text{Confidence Band}_t = \left[ \hat{y}_{t, \, 0.05}, \, \hat{y}_{t, \, 0.95} \right]$$

### 2. Residual Anomaly Z-Score
Systemic imbalance days ("Surplus" vs "Shortage") are detected by calculating normalized residual deviations $e_t = y_t - \hat{y}_t$ over rolling time windows:

$$Z_t = \frac{e_t - \mu_e}{\sigma_e}$$

Anomalies are flagged when $\vert{}Z_t\vert{} \ge 2.0$, signaling abnormal weather shifts or transmission constraints.

### 3. Embedding Cosine-Similarity Guardrail
LLM-generated market thesis candidates $\mathbf{v}_{\text{llm}}$ are embedded via `sentence-transformers` (`all-MiniLM-L6-v2`) and compared against the grounded RAG context vector $\mathbf{u}_{\text{ctx}}$:

$$S_{\cos}(\mathbf{u}_{\text{ctx}}, \mathbf{v}_{\text{llm}}) = \frac{\mathbf{u}_{\text{ctx}} \cdot \mathbf{v}_{\text{llm}}}{\Vert{}\mathbf{u}_{\text{ctx}}\Vert{} \Vert{}\mathbf{v}_{\text{llm}}\Vert{}} \ge 0.85$$

Responses scoring $S_{\cos} < 0.85$ or failing Pydantic schema verification are automatically rejected and re-routed for deterministic fallback synthesis.

---

## ✨ Key Technical Capabilities

1. **Multi-Bidding Zone Ingestion Pipeline (`src/alphagrid/data/`)**:
   * Ingests actual wind/solar generation via **ENTSO-E REST API** and meteorological metrics via **Open-Meteo** across `DE_LU`, `FR`, `NL`, and `DK_1`.
   * Enforces explicit UTC localization across all pandas `DatetimeIndex` objects to prevent offset mismatch errors.
   * Caches raw telemetry locally as Parquet slices in `artifacts/raw_cache/` to minimize external network overhead, featuring a deterministic synthetic data fallback generator.

2. **Probabilistic Forecasting & Optuna Engine (`src/alphagrid/forecasting/`)**:
   * Automated **Optuna hyperparameter optimization** (`tune.py`) evaluating cross-validation Mean Absolute Error (MAE) across LightGBM regressors.
   * Constructs rolling lag features, exponential moving averages, and time-series calendar embeddings.
   * Generates 24-to-48 hour forecasts with prediction intervals and tracks daily residual Z-scores.

3. **Multi-Agent RAG & Debate Architecture (`src/alphagrid/agents/`)**:
   * **Bull vs. Bear Debate (`debate_agent.py`)**: Executes parallel prompt orchestration between a **Bullish Agent** (upside price/demand catalysts) and **Bearish Agent** (downside oversupply catalysts).
   * **Ingestion Agent**: Strips raw HTML tags from energy news RSS feeds before chunking and embedding text into persistent **ChromaDB** collections (`artifacts/chroma/`).
   * **Mathematical Guardrail (`guardrail.py`)**: Enforces structured Pydantic validation (`MarketThesis`), verifies verbatim citation substrings, and validates cosine similarity ($S_{\cos} \ge 0.85$).

4. **Interactive Analytics & PDF Reporting (`src/alphagrid/dashboard/`)**:
   * Built with Streamlit, Plotly, and `fpdf2`.
   * Provides real-time bidding zone switching, interactive prediction band visualization, Optuna study execution inspection, and **one-click Executive PDF Report generation**.

5. **Production Docker Containerization (`Dockerfile`, `docker-compose.yml`)**:
   * Multi-stage containerized architecture bundling the Python dashboard and Ollama LLM service for one-command orchestration (`docker compose up`).

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
├── artifacts/                  # Local model binaries, Parquet cache, & ChromaDB vector store
├── src/
│   └── alphagrid/
│       ├── config.py           # Configuration & .env loader
│       ├── data/               # ENTSO-E, Open-Meteo, & Feature Engineering Store
│       │   ├── entsoe_client.py
│       │   ├── weather_client.py
│       │   ├── time_utils.py
│       │   └── feature_store.py
│       ├── forecasting/        # LightGBM Engine, Predictor, Anomaly Detector, & Optuna Tuner
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── anomalies.py
│       │   └── tune.py
│       ├── llm/                # Local Ollama Client Wrapper
│       │   └── ollama_client.py
│       ├── agents/             # RSS Ingestion, Synthesis, Debate, Guardrail, Orchestrator
│       │   ├── ingestion_agent.py
│       │   ├── guardrail.py
│       │   ├── synthesis_agent.py
│       │   ├── debate_agent.py
│       │   └── orchestrator.py
│       └── dashboard/          # Streamlit Interactive App & PDF Brief Generator
│           ├── app.py
│           └── pdf_report.py
├── tests/                      # Automated Unit & Integration Test Suite (22/22 passing)
│   ├── test_data.py
│   ├── test_predict.py
│   ├── test_anomalies.py
│   ├── test_agents.py
│   ├── test_dashboard.py
│   └── test_enhancements.py
└── pyproject.toml              # Project dependencies, Ruff, & Mypy configurations
```
## 🚀 Quick Start Guide

### Option A: Fast Local Environment Setup (`uv`)

```powershell
# 1. Install dependencies using uv
uv sync

# 2. Add your ENTSO-E API Security Token to .env
ENTSOE_TOKEN=your_entsoe_api_security_token_here

# 3. Launch Streamlit Dashboard
streamlit run src/alphagrid/dashboard/app.py
```
### Option B: Docker Container Deployment
```powershell

# Build and spin up the complete application stack (Dashboard + Ollama)
docker compose up --build
Access the interactive trading portal at http://localhost:8501.
```
### Testing & Code Quality Assurance
```powershell
The codebase adheres to strict software engineering standards, fully verified via automated tests and static analysis:The codebase adheres to strict software engineering standards, fully verified via automated tests and static analysis:PowerShell#

# 1. Execute full unit and integration test suite (22/22 passing)
pytest

# 2. Run static type analysis (0 type errors across 29 modules)
mypy src tests

# 3. Run linter and code formatting checks
ruff check src tests
ruff format --check .se.

📜 License
Distributed under the MIT License. See LICENSE for details.
