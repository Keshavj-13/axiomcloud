# Axiom Cloud AI — AutoML Platform

> A production-grade, full-stack AutoML platform. Upload datasets, train ML models automatically, compare performance on an interactive leaderboard, and deploy models for live predictions.

---

## Platform Overview

Axiom Cloud AI mirrors the core workflow of Google Vertex AI and Kaggle AutoML:

| Feature | Details |
|---|---|
| **AutoML Engine** | Trains 5 models automatically with cross-validation |
| **Task Detection** | Auto-detects classification vs. regression from target column |
| **Model Comparison** | Interactive leaderboard with charts, confusion matrices, ROC curves |
| **Feature Importance** | Visual bar chart of top features per model |
| **Predictions** | REST endpoint for inference on deployed models |
| **Model Export** | Download any model as `.joblib` |
| **Example Datasets** | California Housing, Titanic, Breast Cancer, Iris — all built-in |
| **Adaptive Tuning (New)** | Optional Optuna-based hyperparameter tuning with trial/time budgets |
| **Experiment Registry (New)** | Stores run config, status, best model, and summary metrics per training run |
| **Workspace Search (New)** | Unified dashboard search page across jobs, datasets, and models |
| **EDA Report (New)** | Structured exploratory analysis with correlations, leakage warnings, typing intelligence, and recommendations |

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
| `POST` | `/api/datasets/{id}/clean-and-save` | Create and store cleaned dataset copy |
| `POST` | `/api/load-example/{key}` | Load example dataset |
| `POST` | `/api/train-model` | Start AutoML training job |
| `GET` | `/api/training-status/{job_id}` | Poll training progress |
| `GET` | `/api/training-jobs` | List training jobs |
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
| Datasets | `/dashboard/datasets` | Upload, explore, delete datasets |
| Training | `/dashboard/training` | Configure and launch AutoML jobs |
| Leaderboard | `/dashboard/models` | Compare models, charts, confusion matrix |
| Search | `/dashboard/search` | Cross-entity search for jobs, datasets, and models |
| Predict | `/dashboard/predict` | Run predictions via form UI |
| Deploy | `/dashboard/deploy` | Manage deployed models |

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
