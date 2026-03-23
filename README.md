# Axiom Cloud AI — AutoML Platform

> A production-grade, full-stack AutoML platform for dataset diagnostics, controlled model training (server or local GPU), explainability, deployment, and live inference.

---

## Platform Overview

Axiom Cloud AI mirrors a practical Vertex/Kaggle-style workflow with stronger diagnostics and controlled execution.

### Shipped Features (Current)

#### 1) Data onboarding and profiling
- Upload CSV/Excel datasets.
- Built-in example datasets for instant trials.
- Dataset quality report (missingness, duplicates, outliers, score + recommendations).
- EDA report with chart artifacts and metadata.
- Leakage risk analysis per feature.
- Drift baseline snapshot generation for later monitoring.
- Clean preview and clean-and-save workflow.

#### 2) AutoML training and experiment control
- Auto-detect task type (`classification` / `regression`) when not explicitly set.
- Task-aware model catalog and recommendation flow.
- Hard safety limits enforced in UI + backend:
  - max **5 models** per run,
  - max **5 CV folds**.
- Adaptive hyperparameter tuning with Optuna (trial and time budgets).
- Dataset-aware starting hyperparameters (sensible defaults before tuning).
- Expert mode with optional per-model hyperparameter overrides.
- Experiment registry with run configs, status, best model, and summary metrics.

#### 3) Execution modes
- **Remote mode**: server-side training queue/execution.
- **Local mode**: prepares local job spec, trains on your machine (GPU if available), then syncs results back.
- Local agent download from UI with authenticated request.
- Offline local sync payload support and later re-sync.

#### 4) Model evaluation and explainability
- Leaderboard comparison across trained models.
- Classification visuals: confusion matrix, ROC, CV folds, key metrics.
- Regression visuals: residual/error diagnostics and metric comparisons.
- SHAP and LIME endpoints for model explainability.
- Feature importance extraction for supported estimators.

#### 5) Deployment and inference
- Deploy/undeploy model lifecycle controls.
- Inference sandbox with generated feature template.
- Randomized defaults sampled from training data profile.
- Integer-like numeric fields now produce integer defaults (not float-only noise).
- REST prediction endpoint for programmatic inference.
- Model artifact download (`.joblib`).

#### 6) Security and UX
- Firebase-authenticated API access.
- Responsive dashboard pages for datasets, training, models, predict, deploy, and search.
- Authenticated local-agent command generation for local runs.

---

## Website Use Cases

### 1) Fast baseline AutoML (analyst/data scientist)
Upload a dataset, select target, use auto recommendations, train up to 5 models, and compare results quickly.

### 2) Safe dataset validation before training (ML engineer)
Run quality + leakage + EDA checks first, then decide feature/target readiness before expensive runs.

### 3) Controlled experimentation (team workflows)
Use CV/tuning budgets, store experiment runs, track best model and summary outcomes over time.

### 4) Local compute / GPU-assisted training (hybrid workflow)
Prepare a local job from the web UI, run training on a local machine, and sync metrics/artifacts back to the platform.

### 5) Model review and explainability (stakeholder handoff)
Use leaderboard metrics plus SHAP/LIME outputs to justify model behavior.

### 6) Deployment + inference sandbox (productization)
Deploy selected models, validate inputs in sandbox, and call prediction APIs from external applications.

### 7) Demo/education workflow
Use built-in example datasets to demonstrate complete ML lifecycle without external data prep.

---

## Current Project Goals

Use this as the active backlog. When a goal is fully implemented and verified, move it to the **Platform Overview** table and remove it from this section.

- [ ] **Data Understanding Upgrade**
  - [x] Automatic EDA report generation (distribution plots, correlation summaries)
  - [x] Target leakage checks
  - [x] Feature typing intelligence (ordinal/nominal/datetime-like)
  - [x] Data drift baseline snapshot storage
  - [ ] Pairplot-style feature relationship exploration
  - [ ] Downloadable EDA artifact export (JSON/PDF)

- [ ] **Adaptive AutoML Search**
  - [x] Hyperparameter optimization (Optuna)
  - [x] Time-budget-based tuning
  - [x] Trial-budget tuning controls
  - [ ] Early stopping of weak candidates
  - Progressive resource allocation
  - Metadata-driven model priors (meta-learning)

- [ ] **Feature Engineering Engine**
  - Automatic feature generation (interactions/polynomial/encodings)
  - Feature selection (MI/RFE)
  - Optional dimensionality reduction (PCA)

- [ ] **Experiment Tracking and Reproducibility**
  - [x] Experiment registry (configs + summary metrics + best model)
  - [x] Training UI panel for latest experiment runs
  - [ ] Dataset version linkage in experiment records
  - [ ] Seed and pipeline snapshot tracking
  - [ ] Run-to-run comparison UI

- [ ] **Explainability Deepening**
  - Strong global/local explainability views
  - Feature interaction visualizations
  - Counterfactual explanations

- [ ] **Deployment Intelligence**
  - A/B testing
  - Shadow deployment
  - Rollback mechanism
  - Latency monitoring

- [ ] **Production Monitoring Expansion**
  - Data drift (KS/PSI)
  - Concept drift/performance decay
  - Alerting workflows

- [ ] **Advanced EDA and Discovery**
  - Class imbalance diagnostics with auto-threshold warnings
  - Outlier cluster exploration views
  - Target-feature interaction ranking

- [ ] **Data Contract and Validation Layer**
  - Schema expectation checks before training
  - Constraint validation (ranges, enums, nullability)
  - Contract drift alerts between versions

- [ ] **Dataset Versioning and Lineage**
  - Version every dataset stage
  - Track lineage from raw to transformed assets

- [ ] **Advanced Prediction Interface**
  - Batch predictions
  - Confidence presentation improvements
  - Explain-prediction action in UI
  - CSV bulk inference upload

- [ ] **Performance Optimization**
  - Parallel model training
  - GPU support toggle
  - Preprocessing/pipeline caching

- [ ] **Security and Production Hardening**
  - API authentication (JWT or API keys)
  - Rate limiting
  - Input validation hardening

- [ ] **Advanced UX for Model Ops**
  - Training progress timeline
  - Interactive leaderboard filters
  - Model comparison radar charts
  - Dataset insights dashboard

- [ ] **Strategic Differentiator**
  - Choose and optimize for one identity:
    - Explainability-first AutoML, or
    - Low-data AutoML, or
    - Real-time adaptive models, or
    - Domain-specific AutoML

### Goal Management Rule

1. Do not mark a goal complete until backend, frontend, tests, and docs are all updated.
2. When complete:
   - Move it from **Current Project Goals** to **Platform Overview** as a shipped capability.
   - Remove it from the goals checklist (no duplicates).
3. Keep backlog depth constant:
   - Every time one goal is completed and removed, add one new next-priority goal to this section.

---

## Architecture

```
axiom-cloud-ai/
├── backend/                   # FastAPI + Python ML backend
│   ├── app/
│   │   ├── main.py            # FastAPI app entry point
│   │   ├── api/
│   │   │   ├── datasets.py    # /upload-dataset, /datasets
│   │   │   ├── training.py    # /train-model, /training-status
│   │   │   ├── models.py      # /models, /deploy, /download
│   │   │   ├── predictions.py # /predict
│   │   │   └── metrics.py     # /metrics/{job_id}
│   │   ├── core/
│   │   │   ├── config.py      # Pydantic settings
│   │   │   └── database.py    # SQLAlchemy engine
│   │   ├── ml/
│   │   │   ├── pipeline.py    # AutoML pipeline (core engine)
│   │   │   └── datasets.py    # Example dataset loaders
│   │   ├── models/
│   │   │   └── db_models.py   # SQLAlchemy ORM models
│   │   └── schemas/
│   │       └── schemas.py     # Pydantic API schemas
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env
│
├── frontend/                  # Next.js 14 + Tailwind frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx       # Landing page
│   │   │   └── dashboard/
│   │   │       ├── page.tsx          # Dashboard home
│   │   │       ├── datasets/page.tsx # Dataset upload + explorer
│   │   │       ├── training/page.tsx # AutoML training config
│   │   │       ├── models/page.tsx   # Leaderboard + charts
│   │   │       ├── predict/page.tsx  # Prediction interface
│   │   │       └── deploy/page.tsx   # Deployment management
│   │   ├── components/
│   │   │   └── layout/Sidebar.tsx
│   │   ├── lib/api.ts         # Axios API client
│   │   └── types/index.ts     # TypeScript types
│   ├── package.json
│   ├── tailwind.config.js
│   └── Dockerfile
│
├── database/
│   └── init.sql               # PostgreSQL schema
├── docker-compose.yml
└── README.md
```

---

## Quick Start (Local Dev — No Docker)

### Prerequisites
- Python 3.10 (recommended for pinned ML dependencies)
- Node.js 18+
- pip

### 0. One-command local workflow (recommended)

From the project root:

```bash
npm install
npm run dev
```

This starts both services concurrently with labeled logs:
- `BACKEND` → FastAPI on `http://localhost:8000` with reload
- `FRONTEND` → Next.js on `http://localhost:3000` with hot reload

If either service exits, the combined dev session stops immediately and the failure is visible in labeled logs.

### 1. Clone the repository
```bash
git clone https://github.com/keshavj-13/axiom-cloud-ai.git
cd axiom-cloud-ai
```

### 2. Backend Setup

```bash
cd backend

# Option A (recommended): Conda + Python 3.10
conda create -n sigmacld310 python=3.10 -y
conda activate sigmacld310

# Option B: venv
# python3.10 -m venv venv
# source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate          # Windows

# Install dependencies (binds install to the active Python interpreter)
python -m pip install -r requirements.txt

# Configure environment (defaults to SQLite, no Redis needed for local dev)
cp .env.example .env

# Start the API server (uses the same interpreter/environment)
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> After the backend environment is set up once, return to project root and use `npm run dev` for daily development.

The API is now running at **http://localhost:8000**
Swagger docs at **http://localhost:8000/api/docs**

### 3. Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install --legacy-peer-deps

# Start the dev server
npm run dev
```

The frontend is now running at **http://localhost:3000**

---

## 🐳 Docker Setup (Full Stack)

```bash
# From project root
docker-compose up --build

# Services:
# Frontend:  http://localhost:3000
# Backend:   http://localhost:8000
# API Docs:  http://localhost:8000/api/docs
# Postgres:  localhost:5432
# Redis:     localhost:6379
```

### Production Deploy (VPS/Cloud VM)

Use environment overrides so frontend points to your real API URL:

```bash
cp .env.production.example .env.production
# edit .env.production values
docker compose --env-file .env.production up -d --build
```

Detailed steps: see [`DEPLOYMENT.md`](DEPLOYMENT.md)

---

## AutoML Pipeline

The pipeline in `backend/app/ml/pipeline.py` performs:

### 1. Task Detection
```python
# Auto-detects classification vs regression:
# - dtype is object/bool → classification
# - ≤ 20 unique values AND < 5% of total rows → classification
# - Otherwise → regression
```

### 2. Preprocessing
- **Missing values**: Median imputation (numeric), Mode imputation (categorical)
- **Feature encoding**: OneHotEncoder for categoricals (≤50 unique values)
- **Scaling**: StandardScaler for all numeric features
- **Train/test split**: 80/20 by default, stratified for classification
- Drops high-cardinality categoricals (>50 unique values)

### 3. Models Trained

**Classification:**
| Model | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Random Forest | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| Gradient Boosting | scikit-learn |

**Regression:**
| Model | Library |
|---|---|
| Ridge Regression | scikit-learn |
| Random Forest | scikit-learn |
| XGBoost | xgboost |
| LightGBM | lightgbm |
| Gradient Boosting | scikit-learn |

### 4. Evaluation
- **Classification**: Accuracy, F1, ROC-AUC, Confusion Matrix, ROC Curve
- **Regression**: RMSE, MAE, R²
- **Cross-Validation**: Stratified K-Fold (default 5 folds) on all models
- **Feature importance**: Extracted from tree models; abs(coef) for linear models

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload-dataset` | Upload CSV/Excel file |
| `GET` | `/api/datasets` | List all datasets |
| `GET` | `/api/datasets/{id}/quality-report` | Dataset quality report + recommendations |
| `GET` | `/api/datasets/{id}/eda-report` | Structured exploratory data analysis report |
| `GET` | `/api/datasets/{id}/leakage-report` | Target leakage risk report |
| `GET` | `/api/datasets/{id}/drift-baseline` | Baseline distribution snapshot for drift monitoring |
| `GET` | `/api/datasets/{id}/clean-preview` | Non-destructive auto-clean preview |
| `GET` | `/api/datasets/{id}/analytics-report` | Combined analytics report (EDA + model evaluation charts) |
| `POST` | `/api/datasets/{id}/clean-and-save` | Create and store cleaned dataset copy |
| `POST` | `/api/load-example/{key}` | Load example dataset |
| `POST` | `/api/train-model` | Start AutoML training job |
| `GET` | `/api/training-status/{job_id}` | Poll training progress |
| `GET` | `/api/training-jobs` | List training jobs |
| `GET` | `/api/training/model-catalog` | Task-aware model catalog + model metadata |
| `GET` | `/api/training/local-job-spec/{job_id}` | Fetch local training spec for local execution mode |
| `POST` | `/api/training/local-sync` | Sync local training results to backend |
| `GET` | `/api/training/local-agent/download` | Download `local_agent.py` |
| `GET` | `/api/experiments` | List experiment runs |
| `GET` | `/api/experiments/{run_id}` | Get single experiment run |
| `GET` | `/api/models` | List trained models |
| `GET` | `/api/models/{id}/shap` | SHAP explanation payload |
| `GET` | `/api/models/{id}/lime` | LIME explanation payload |
| `GET` | `/api/models/{id}/monitoring` | Model health + optional drift report |
| `POST` | `/api/models/{id}/deploy` | Deploy model |
| `GET` | `/api/models/{id}/download` | Download model (.joblib) |
| `POST` | `/api/predict` | Run inference |
| `GET` | `/api/metrics/{job_id}` | Get comparison metrics |
| `GET` | `/api/health` | Health check |

### Example: Train a model via API
```bash
# 1. Upload dataset
curl -X POST http://localhost:8000/api/upload-dataset \
  -F "file=@data.csv" \
  -F "name=MyDataset"

# 2. Start training
curl -X POST http://localhost:8000/api/train-model \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": 1, "target_column": "price", "cv_folds": 5, "enable_tuning": true, "tuning_trials": 16, "tuning_time_budget_sec": 180}'

# 3. Check status
curl http://localhost:8000/api/training-status/{job_id}

# 4. Get metrics
curl http://localhost:8000/api/metrics/{job_id}

# 5. Run prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": 1, "features": {"feature1": 5.2, "feature2": "value"}}'
```

---

## Dashboard Pages

| Page | Route | Description |
|---|---|---|
| Landing | `/` | Marketing page with platform overview |
| Dashboard | `/dashboard` | Stats, recent jobs, quick actions |
| Datasets | `/dashboard/datasets` | Upload, clean, profile, leakage/EDA/analytics exploration |
| Training | `/dashboard/training` | Configure remote/local runs, expert mode, tuning, local-agent flow |
| Leaderboard | `/dashboard/models` | Compare models, evaluation visuals, diagnostics, explainability hooks |
| Search | `/dashboard/search` | Cross-entity search for jobs, datasets, and models |
| Predict | `/dashboard/predict` | Inference sandbox with dataset-derived feature defaults |
| Deploy | `/dashboard/deploy` | Deploy lifecycle and endpoint-style inference testing |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Recharts |
| **Backend** | FastAPI, Python 3.11, Pydantic v2 |
| **ML** | scikit-learn, XGBoost, LightGBM, joblib |
| **Database** | SQLite (dev) / PostgreSQL (prod) |
| **ORM** | SQLAlchemy 2.0 |
| **Background** | FastAPI BackgroundTasks / Celery + Redis (prod) |
| **Container** | Docker + Docker Compose |
| **Storage** | Local filesystem (extendable to S3) |

---

## Resume Highlights

This project demonstrates:
- **Full-stack ML system design** — API → training pipeline → frontend in production architecture
- **AutoML concepts** — task detection, preprocessing pipelines, multi-model comparison, CV
- **Clean REST API design** — proper schemas, error handling, background jobs
- **Modern React** — hooks, TypeScript, real-time polling, drag & drop
- **Production practices** — Docker, environment configs, health checks, logging
- **ML engineering** — sklearn Pipelines, feature encoding, model serialization, evaluation metrics

---

## License

MIT © Axiom Cloud AI
