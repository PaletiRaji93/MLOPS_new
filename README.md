# Energy Market Forecasting MLOps Pipeline

End-to-end MLOps platform for predicting **Day Ahead Market Clearing Price (DAM MCP)** in the Indian energy market. Covers the full lifecycle: data ingestion, validation, feature engineering, model training, evaluation, automated promotion, serving, monitoring, and CI/CD.

## Documentation — follow in order

New to this project? Work through the guides in `docs/` in sequence:

1. **`docs/1_MLOps_Installation_Guide.docx`** — install every tool the stack needs (Python, Git, Docker, DVC, MLflow, Flask, Feast, minikube, Argo CD, ruff, ...) on Windows or macOS, each with a verification step. **Start here on a fresh machine.**
2. **`docs/2_DEMO_MASTER_RUNBOOK.md`** — the end-to-end demo: ingestion → validation → features → train/evaluate/promote → serving → monitoring, plus the CI/CD → GitOps walkthrough.
3. **`docs/3_ARGOCD_K8S_INSTALL_RUNBOOK.md`** — one-time setup of the local minikube cluster + Argo CD that the runbook's GitOps section deploys to.

> Already installed the tools? Skip to doc 2 (run the installation guide's *Verification Checklist* first to confirm nothing is missing). Doc 3 is only needed the first time you stand up the Kubernetes / Argo CD side.

## Architecture

```
                              DATA SOURCES
      ┌──────────────────┬───────────────────┬──────────────────┐
      │   IEX India      │   Open-Meteo API  │   NPP (CEA)     │
      │   (Selenium)     │   (REST API)      │   (Playwright)  │
      │   DAM + RTM      │   65 Indian cities│   DGR PDFs      │
      └────────┬─────────┴─────────┬─────────┴────────┬────────┘
               │                   │                   │
               ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  BRONZE (Raw)                                        data/raw/  │
│  ├── dam/year=YYYY/month=MM/date=YYYY-MM-DD/dam.csv             │
│  ├── rtm/year=YYYY/month=MM/date=YYYY-MM-DD/rtm.csv             │
│  ├── weather/{City_Name}.csv                                     │
│  ├── generation/raw/{year}/{YYYY-MM-DD}.pdf → csv/dgr_{FY}.csv  │
│  └── calendar/calendar.csv                                       │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  SILVER (Cleaned)                              data/processed/  │
│  Parquet files partitioned by date                               │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                    Great Expectations
                    (row-level validation)
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  VALIDATED                                    data/validated/   │
│  ├── *_valid/    (clean rows)                                    │
│  └── *_invalid/  (bad rows + failure reasons)                    │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                          data_prep.py
                    (join, lag, rolling, cyclical)
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  GOLD (Features)        data/features/                          │
│  ~30+ features per 15-min block                                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
              ┌────────────────┤
              ▼                ▼
┌───────────────────┐  ┌──────────────────────────────────────────┐
│  Feast Feature    │  │  DVC Pipeline (dvc.yaml)                 │
│  Store            │  │  prepare_feast_data → train → evaluate   │
│  (offline Parquet)│  │                                → promote │
└───────────────────┘  └──────────────────────┬───────────────────┘
                                              │
                           ┌──────────────────┤
                           ▼                  ▼
                    ┌─────────────┐   ┌──────────────┐
                    │  MLflow     │   │  model.pkl   │
                    │  Tracking + │   │  metrics.json│
                    │  Registry   │   │  eval_metrics│
                    └──────┬──────┘   └──────────────┘
                           │
              app_config/model_version.json committed to main
                           │
              ┌────────────┼──────────────────────────┐
              ▼            ▼                          ▼
┌───────────────┐  ┌────────────┐  ┌──────────┐  ┌───────────────────────┐
│  CI/CD        │  │  Docker    │  │  ArgoCD  │  │  K8s (minikube)       │
│  GitHub       │──│  (GHCR)   │──│  GitOps  │──│  Flask + Prometheus   │
│  Actions      │  │           │  │          │  │  + Grafana + MLflow   │
└───────────────┘  └───────────┘  └──────────┘  └───────────────────────┘
```

## Tech Stack

| Layer | Tools |
|---|---|
| Data Ingestion | Selenium (IEX DAM/RTM), Open-Meteo API (weather), Playwright + pdfplumber (generation PDFs) |
| Data Validation | Great Expectations (row-level valid/invalid split with failure reasons) |
| Data Versioning | DVC (data files tracked via `.dvc` pointer files, Git tracks code) |
| Feature Engineering | Pandas (Bronze → Silver → Gold medallion architecture) |
| Feature Store | Feast (offline store, Parquet) |
| Training | Scikit-learn, XGBoost, ARIMA, Prophet, Exponential Smoothing |
| Experiment Tracking | MLflow (tracking, registry, artifacts) |
| Pipeline Orchestration | DVC (4-stage pipeline: prepare → train → evaluate → promote) |
| Serving | Flask + Gunicorn (4 workers) |
| Data Lineage | OpenLineage + Marquez (end-to-end lineage graph across all pipeline stages) |
| Monitoring | Prometheus (metrics), Grafana (dashboards + alerts), Evidently (drift) |
| CI/CD | GitHub Actions (lint, test, train, build, deploy) |
| Infrastructure | Docker, Kubernetes (minikube), ArgoCD (GitOps) |
| Container Registry | GitHub Container Registry (GHCR) |

## Data Sources

| Source | Method | Output | Schedule |
|---|---|---|---|
| **DAM** (Day Ahead Market) | Selenium scraping from IEX India | 96 sessions/day (15-min blocks): MCP, MCV, Purchase/Sell Bids | Daily |
| **RTM** (Real Time Market) | Selenium scraping from IEX India | Same structure as DAM | Daily |
| **Weather** | Open-Meteo REST API | Hourly data for 65 Indian cities: temperature, humidity, windspeed, cloud cover, rainfall | Configurable |
| **Generation** (DGR) | Playwright PDF download from NPP + pdfplumber parsing | Plant-level generation capacity (thermal, hydro, nuclear, solar, wind) | Daily |
| **Calendar** | Static CSV | Holidays, festivals, IPL matches, elections, weekends | Manual update |

Run ingestion scripts:
```bash
python src/1_ingestion/1_ingest_dam_rtm_local.py --date 2025-06-15 --force
python src/1_ingestion/3_ingest_weather_local.py --start 2025-06-01 --end 2025-06-30
python src/1_ingestion/2_ingest_generation_local.py --start 2025-06-15 --end 2025-06-15
```

## Data Validation (Great Expectations)

Each data source has its own expectation suite in `gx/expectations/`:

| Suite | Rules | Examples |
|---|---|---|
| `weather_suite.json` | 13 rules | Temperature -50 to 60, humidity 0-100, rain requires clouds (95% tolerance), clear sky = no rainfall |
| `dam_suite.json` | 12 rules | (Datetime, Session ID) uniqueness, MCP 0-20000, Session ID 1-96, no nulls |
| `rtm_suite.json` | 12 rules | Same pattern as DAM for RTM data |
| `generation_suite.json` | Rules | Capacity bounds, plant name validation |
| `calendar_suite.json` | Rules | Date format, boolean flags, no duplicates |

Validation splits data into **valid** and **invalid** DataFrames with per-row failure reasons:
```bash
python src/2_validation/5_gx_validate_local.py
```

## Data Versioning (DVC)

Raw data and feature files are tracked by DVC, not Git. The repo stores small `.dvc` pointer files (MD5 hash + size) while actual data lives in the local DVC cache.

```
data/raw/weather.dvc      → tracks data/raw/weather/ (65 city CSVs)
data/raw/dam.dvc           → tracks data/raw/dam/ (daily market data)
data/raw/rtm.dvc           → tracks data/raw/rtm/ (real-time market data)
data/raw/generation.dvc    → tracks data/raw/generation/ (PDFs + CSVs)
data/raw/calendar.dvc      → tracks data/raw/calendar/
```

After cloning, restore data from DVC cache:
```bash
dvc checkout
```

## DVC Pipeline

```
prepare_feast_data  →  train  →  evaluate  →  promote
       │                 │           │            │
       ▼                 ▼           ▼            ▼
  features.parquet   model.pkl   eval_metrics  promote_report
                     metrics.json   .json         .json
```

Run the full pipeline:
```bash
dvc repro
```

View metrics:
```bash
dvc metrics show
```

## Data Lineage (Marquez + OpenLineage)

Every pipeline stage emits OpenLineage events to Marquez, building an end-to-end lineage graph:

```
IEX India ──► ingest_dam_rtm ──► data/raw/dam, data/raw/rtm
Open-Meteo ──► ingest_weather ──► data/raw/weather
NPP (CEA) ──► ingest_generation ──► data/raw/generation
                    │
                    ▼
data/processed/* ──► validate_{dataset} ──► data/validated/*/valid
                                        ──► data/validated/*/invalid
                    │
                    ▼
features ──► prepare_feast_data ──► feast/march_2025_features.parquet
                    │
                    ▼
feast features ──► train ──► models/model.pkl + metrics.json
                    │
                    ▼
model.pkl ──► evaluate ──► eval_metrics.json
                    │
                    ▼
eval_metrics ──► promote ──► promote_report.json + MLflow champion alias
```

Start Marquez (opt-in, runs as its own separate stack):
```bash
docker compose -f marquez-docker-compose.yml up -d
```

| Service | Port | URL |
|---|---|---|
| Marquez Web UI | 3001 | http://localhost:3001 |
| Marquez API | 5002 | http://localhost:5002 |

Lineage is non-blocking: if Marquez is not running, all pipeline scripts continue normally without emitting events.

## Quick Start

### Prerequisites

- Python 3.12+
- Docker & Docker Compose

### 1. Clone and install

```bash
git clone https://github.com/srija77/my-mlops-project.git
cd my-mlops-project
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
dvc checkout               # restore data files from DVC cache
```

### 2. Start the stack

```bash
cp .env.example .env       # fill in API_KEY and GRAFANA_PASSWORD
docker compose up -d
```

This starts 5 services:

| Service | Port | URL |
|---|---|---|
| App (Flask) | 8000 | http://localhost:8000 |
| MLflow | 5000 | http://localhost:5000 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |
| Postgres | 5432 | — |

To also start data lineage tracking (runs from its own compose file):
```bash
docker compose -f marquez-docker-compose.yml up -d   # Marquez Web (3001) + API (5002)
```

### 3. Run the training pipeline

```bash
dvc repro
```

Or trigger via CI:
- Go to **Actions** > **ML CI/CD Pipeline** > **Run workflow** > check **Run full DVC training pipeline**

## Models

Five models compete on each training run:

| Model | Type |
|---|---|
| ARIMA | Time series (univariate) |
| Exponential Smoothing | Time series (Holt-Winters) |
| Prophet | Time series (Facebook) |
| Gradient Boosting | Tree-based (sklearn) |
| XGBoost | Tree-based |

The best model (by RMSE) is registered in MLflow. The evaluate stage gates promotion — only models with `passed_eval=True` can become the production champion.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI with prediction form and dashboard |
| `/predict` | POST | Single prediction (JSON or form data) |
| `/batch-predict` | POST | Upload CSV, get predictions + drift report |
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe (503 until model loaded) |
| `/reload` | POST | Hot-reload model from MLflow |
| `/metrics` | GET | Prometheus metrics |
| `/drift` | GET | Latest drift detection summary |
| `/deployment` | GET | Deployment metadata |

## CI/CD Pipeline

Two delivery paths:

**Code changes** (automatic on push to main):
```
push → test (lint + pytest) → build Docker images → push to GHCR → update K8s manifests → ArgoCD syncs
```

**Model updates** (manual trigger):
```
workflow_dispatch → test → dvc repro (train → evaluate → promote) → commit model_version.json
    → triggers: build → push to GHCR → update K8s manifests → ArgoCD syncs
```

**Automated retraining on drift** (scheduled — `retrain-on-drift.yml`):
```
cron → drift.py on recent data → if dataset_drift: dispatch the train job above
    → train → evaluate → promote → CD (same chain as model updates)
```

## Monitoring & Alerting

### Prometheus Metrics
- `inference_requests_total` — total prediction requests
- `inference_errors_total` — failed predictions
- `inference_latency_seconds` — response time histogram
- `model_loaded` — whether model is serving (1/0)
- `drift_detected` — whether data drift is active (1/0)

### Grafana Alert Rules
- Data drift detected
- High error rate
- Model not loaded
- High inference latency
- No inference traffic

> Prometheus `rate()`/`histogram_quantile()` panels only show data while the app is
> actively serving. If the graphs look empty, generate sustained load:
> `python scripts/generate_traffic.py --seconds 300 --rps 5`

## Project Structure

```
├── app/                        # Flask serving app (Gunicorn, 4 workers)
│   ├── app.py
│   ├── db.py
│   ├── gunicorn.conf.py        # Prometheus multiprocess config
│   ├── templates/
│   └── static/
├── src/
│   ├── 1_ingestion/            # Data source connectors
│   │   ├── 1_ingest_dam_rtm_local.py     # Selenium → IEX India (DAM/RTM)
│   │   ├── 2_ingest_generation_local.py  # Playwright → NPP DGR PDFs
│   │   └── 3_ingest_weather_local.py     # Open-Meteo API → 65 cities
│   ├── 2_validation/           # Great Expectations validators
│   │   ├── 1_validate_dam.py
│   │   ├── 2_validate_rtm.py
│   │   ├── 3_validate_weather.py
│   │   ├── 4_validate_calendar.py
│   │   └── 5_gx_validate_local.py        # Orchestrator (runs all validators)
│   ├── 3_feature_engineering/  # Gold features + Feast online/offline store
│   │   ├── 1_build_features.py            # join + lag + rolling + cyclical
│   │   ├── 2_prepare_feast_data.py
│   │   ├── 3_refresh_online_store.py
│   │   └── 4_query_online_store.py
│   ├── 4_training/1_train.py              # Train 5 models, log to MLflow
│   ├── 5_evaluation/1_evaluate.py         # Evaluate + gate promotion
│   ├── 6_promotion/1_promote_model.py     # Compare & promote champion
│   ├── 7_prediction/1_predict.py          # CLI / batch prediction (online store)
│   ├── 8_monitoring/1_create_reference_data.py  # drift reference baseline
│   ├── drift.py                # Evidently drift detection
│   └── lineage.py              # OpenLineage client (emits events to Marquez)
├── gx/
│   └── expectations/           # GX expectation suite JSONs (per data source)
├── my_feature_store/           # Feast feature repo
│   └── feature_repo/
│       ├── feature_definitions.py
│       ├── feature_store.yaml
│       └── prepare_feast_data.py
├── data/
│   ├── raw/                    # Bronze (tracked by DVC)
│   │   ├── dam.dvc, rtm.dvc, weather.dvc, generation.dvc, calendar.dvc
│   │   └── .gitignore
│   ├── processed/              # Silver (Parquet)
│   ├── validated/              # Valid/invalid split
│   └── features/               # Gold features (tracked by DVC)
├── models/                     # Trained model (tracked by DVC pipeline)
├── app_config/
│   └── model_version.json      # Current production model (triggers CD)
├── drift_baselines/            # reference_data.parquet for drift checks
├── monitoring/
│   ├── prometheus.yml
│   ├── drift_summary.json      # latest drift verdict (exposed to Prometheus)
│   └── provisioning/           # Grafana datasources, dashboards, alerts
├── k8s/
│   ├── deployment.yaml         # App + Prometheus + Grafana + Postgres
│   ├── namespace.yaml
│   └── argocd-app.yaml         # GitOps auto-sync (gmr-mlops app)
├── scripts/
│   ├── populate_raw_data.py    # load bundled March 2025 demo data
│   └── generate_traffic.py     # sustained load generator for monitoring
├── .github/workflows/
│   ├── ml-ci.yml               # test → build → deploy (+ manual train)
│   └── retrain-on-drift.yml    # scheduled drift check → auto-retrain
├── dvc.yaml                    # DVC pipeline (build_features frozen → prepare → train → evaluate → promote)
├── dvc.lock                    # Locked pipeline hashes
├── Dockerfile                  # App image (Gunicorn)
├── Dockerfile.train            # Training / MLflow image
├── docker-compose.yml          # Local dev stack
├── marquez-docker-compose.yml  # Marquez lineage stack (separate)
└── requirements.txt
```

## Features (30+)

- **Market**: DAM/RTM prices, volumes, bid imbalance, MCP spread
- **Lag & Rolling**: 1d/2d/7d lags, 4h/24h rolling mean/std
- **Weather**: Temperature, humidity, windspeed, cloud cover, rainfall
- **Calendar**: Hour/day cyclical encoding, weekend, peak hour, IPL matches, festivals, elections
- **Generation**: Thermal/hydro/solar/wind capacity from DGR reports
