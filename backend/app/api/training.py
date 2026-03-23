"""
Axiom Cloud AI - Training API Router
Launches AutoML training jobs (sync for simplicity, async with Celery in prod).
"""
import uuid
import logging
import os
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.db_models import Dataset, TrainingJob, TrainedModel, ExperimentRun
from app.schemas.schemas import (
    TrainingConfig,
    TrainingJobResponse,
    ExperimentRunResponse,
    TrainingLaunchResponse,
    LocalTrainingSyncPayload,
)
from app.ml.dataset_profiling import build_drift_baseline_snapshot

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory progress tracking (use Redis in production)
_job_progress: dict = {}
MAX_TRAINING_ROWS = 50_000
MAX_MODELS_PER_RUN = 5
MAX_CV_FOLDS = 5

CLASSIFICATION_MODEL_NAMES = [
    "Logistic Regression",
    "Linear SVM",
    "RBF SVM",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "AdaBoost",
    "XGBoost",
    "LightGBM",
    "Gradient Boosting",
    "MLP Neural Net",
    "Deep MLP Neural Net",
    "Graph Neural Network (experimental)",
]

REGRESSION_MODEL_NAMES = [
    "Linear Regression",
    "Linear SVR",
    "RBF SVR",
    "KNN",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "AdaBoost",
    "XGBoost",
    "LightGBM",
    "Gradient Boosting",
    "MLP Neural Net",
    "Deep MLP Neural Net",
    "Graph Neural Network (experimental)",
]

MODEL_HYPERPARAMETERS = {
    "Logistic Regression": {
        "C": {"type": "number", "min": 0.001, "max": 20.0, "step": 0.001, "default": 1.0},
    },
    "Linear Regression": {
        "alpha": {"type": "number", "min": 0.001, "max": 20.0, "step": 0.001, "default": 1.0},
    },
    "KNN": {
        "n_neighbors": {"type": "integer", "min": 1, "max": 50, "step": 1, "default": 7},
    },
    "Decision Tree": {
        "max_depth": {"type": "integer", "min": 2, "max": 30, "step": 1, "default": 12},
        "min_samples_split": {"type": "integer", "min": 2, "max": 20, "step": 1, "default": 2},
    },
    "Random Forest": {
        "n_estimators": {"type": "integer", "min": 80, "max": 500, "step": 10, "default": 180},
        "max_depth": {"type": "integer", "min": 3, "max": 30, "step": 1, "default": 14},
        "min_samples_split": {"type": "integer", "min": 2, "max": 20, "step": 1, "default": 2},
        "min_samples_leaf": {"type": "integer", "min": 1, "max": 10, "step": 1, "default": 1},
    },
    "Extra Trees": {
        "n_estimators": {"type": "integer", "min": 80, "max": 500, "step": 10, "default": 180},
        "max_depth": {"type": "integer", "min": 3, "max": 30, "step": 1, "default": 14},
    },
    "AdaBoost": {
        "n_estimators": {"type": "integer", "min": 50, "max": 500, "step": 10, "default": 180},
        "learning_rate": {"type": "number", "min": 0.01, "max": 2.0, "step": 0.01, "default": 0.1},
    },
    "XGBoost": {
        "n_estimators": {"type": "integer", "min": 80, "max": 500, "step": 10, "default": 220},
        "max_depth": {"type": "integer", "min": 3, "max": 20, "step": 1, "default": 8},
        "learning_rate": {"type": "number", "min": 0.005, "max": 0.5, "step": 0.005, "default": 0.06},
        "subsample": {"type": "number", "min": 0.5, "max": 1.0, "step": 0.01, "default": 0.85},
        "colsample_bytree": {"type": "number", "min": 0.5, "max": 1.0, "step": 0.01, "default": 0.85},
    },
    "LightGBM": {
        "n_estimators": {"type": "integer", "min": 80, "max": 500, "step": 10, "default": 220},
        "num_leaves": {"type": "integer", "min": 16, "max": 256, "step": 1, "default": 64},
        "learning_rate": {"type": "number", "min": 0.005, "max": 0.5, "step": 0.005, "default": 0.06},
        "subsample": {"type": "number", "min": 0.5, "max": 1.0, "step": 0.01, "default": 0.9},
        "colsample_bytree": {"type": "number", "min": 0.5, "max": 1.0, "step": 0.01, "default": 0.9},
    },
    "Gradient Boosting": {
        "n_estimators": {"type": "integer", "min": 60, "max": 500, "step": 10, "default": 180},
        "learning_rate": {"type": "number", "min": 0.005, "max": 0.5, "step": 0.005, "default": 0.05},
        "max_depth": {"type": "integer", "min": 2, "max": 12, "step": 1, "default": 4},
        "subsample": {"type": "number", "min": 0.5, "max": 1.0, "step": 0.01, "default": 0.9},
    },
    "Linear SVM": {
        "C": {"type": "number", "min": 0.01, "max": 50.0, "step": 0.01, "default": 1.0},
    },
    "RBF SVM": {
        "C": {"type": "number", "min": 0.01, "max": 50.0, "step": 0.01, "default": 1.0},
        "gamma": {"type": "number", "min": 0.0001, "max": 5.0, "step": 0.0001, "default": 0.1},
    },
    "Linear SVR": {
        "C": {"type": "number", "min": 0.01, "max": 50.0, "step": 0.01, "default": 1.0},
    },
    "RBF SVR": {
        "C": {"type": "number", "min": 0.01, "max": 50.0, "step": 0.01, "default": 1.0},
        "gamma": {"type": "number", "min": 0.0001, "max": 5.0, "step": 0.0001, "default": 0.1},
    },
    "MLP Neural Net": {
        "alpha": {"type": "number", "min": 0.000001, "max": 0.1, "step": 0.000001, "default": 0.0001},
        "max_iter": {"type": "integer", "min": 100, "max": 1000, "step": 10, "default": 320},
    },
    "Deep MLP Neural Net": {
        "alpha": {"type": "number", "min": 0.000001, "max": 0.1, "step": 0.000001, "default": 0.0001},
        "max_iter": {"type": "integer", "min": 120, "max": 1500, "step": 10, "default": 420},
    },
}


def _model_catalog_payload() -> dict:
    classification = list(CLASSIFICATION_MODEL_NAMES)
    regression = list(REGRESSION_MODEL_NAMES)
    union = sorted(set(classification + regression))
    details = {}
    for name in union:
        is_nn = "Neural" in name
        is_gnn = "Graph Neural" in name
        is_svm = "SVM" in name or "SVR" in name
        is_heavy = is_nn or is_gnn or is_svm
        details[name] = {
            "family": (
                "gnn" if is_gnn else
                "neural" if is_nn else
                "svm" if is_svm else
                "tree" if ("Forest" in name or "Boost" in name or "Tree" in name) else
                "linear"
            ),
            "cost_tier": "high" if is_heavy else "medium",
            "warning": (
                "This option can significantly increase training time."
                if is_heavy else None
            ),
            "experimental": bool(is_gnn),
            "hyperparameters": MODEL_HYPERPARAMETERS.get(name, {}),
        }
    return {
        "classification": classification,
        "regression": regression,
        "all": union,
        "details": details,
    }


def _read_dataset_file(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def run_training(job_id: str, run_id: str, dataset_path: str, config: dict, db_url: str):
    """Background training function."""
    from app.ml.pipeline import AutoMLPipeline
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        # Update job status
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        run = db.query(ExperimentRun).filter(ExperimentRun.run_id == run_id).first()
        if not job:
            return
        job.status = "running"
        if run:
            run.status = "running"
        db.commit()

        def update_progress(progress: int, message: str = ""):
            _job_progress[job_id] = {"progress": progress, "message": message}
            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if job:
                job.progress = progress
                db.commit()

        # Load dataset
        df = _read_dataset_file(dataset_path)

        # Run AutoML
        pipeline = AutoMLPipeline(job_id=job_id, progress_callback=update_progress)
        results = pipeline.train_all_models(
            df=df,
            target_column=config["target_column"],
            task_type=config.get("task_type"),
            test_size=config.get("test_size", 0.2),
            cv_folds=config.get("cv_folds", 5),
            models_to_train=config.get("models_to_train"),
            enable_tuning=config.get("enable_tuning", False),
            tuning_trials=config.get("tuning_trials", 12),
            tuning_time_budget_sec=config.get("tuning_time_budget_sec", 180),
            model_hyperparams=config.get("model_hyperparams"),
        )

        # Update job task_type
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        job.task_type = results["task_type"]

        # Save model results
        for model_result in results["models"]:
            if "error" in model_result:
                continue

            model_metrics = model_result.get("metrics") or {}
            model_metrics = {
                **model_metrics,
                "storage_origin": "remote",
                "execution_mode": "remote",
            }

            trained_model = TrainedModel(
                job_id=job_id,
                model_name=model_result["model_name"],
                model_type=model_result["model_type"],
                task_type=results["task_type"],
                file_path=model_result.get("file_path"),
                accuracy=model_result.get("accuracy"),
                f1_score=model_result.get("f1_score"),
                roc_auc=model_result.get("roc_auc"),
                rmse=model_result.get("rmse"),
                mae=model_result.get("mae"),
                r2_score=model_result.get("r2_score"),
                metrics=model_metrics,
                feature_importance=model_result.get("feature_importance"),
                confusion_matrix=model_result.get("confusion_matrix"),
                roc_curve_data=model_result.get("roc_curve_data"),
                cv_scores=model_result.get("cv_scores"),
                training_time=model_result.get("training_time"),
            )
            db.add(trained_model)

        best_score = None
        best_model_name = None
        metric_name = "accuracy" if results.get("task_type") == "classification" else "r2_score"
        successful_models = [m for m in results["models"] if "error" not in m]
        if successful_models:
            ranked = sorted(
                successful_models,
                key=lambda m: (m.get(metric_name) if m.get(metric_name) is not None else -1e9),
                reverse=True,
            )
            best_model_name = ranked[0].get("model_name")
            best_score = ranked[0].get(metric_name)

        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.progress = 100
        if run:
            run.status = "completed"
            run.task_type = results.get("task_type")
            run.best_model_name = best_model_name
            run.best_score = float(best_score) if best_score is not None else None
            run.summary_metrics = {
                "primary_metric": metric_name,
                "best_model": best_model_name,
                "best_score": best_score,
                "models_trained": len(successful_models),
                "failed_models": len(results["models"]) - len(successful_models),
            }
            run.completed_at = datetime.utcnow()
        db.commit()
        logger.info(f"Training job {job_id} completed")

    except Exception as e:
        logger.exception(f"Training job {job_id} failed: {e}")
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        run = db.query(ExperimentRun).filter(ExperimentRun.run_id == run_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
        if run:
            run.status = "failed"
            run.error_message = str(e)
            run.completed_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()


@router.post("/train-model", response_model=TrainingLaunchResponse)
async def train_model(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Launch an AutoML training job. Requires Firebase authentication."""
    # Validate dataset
    dataset = db.query(Dataset).filter(Dataset.id == config.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate target column
    df_cols = [c["name"] for c in (dataset.columns_info or [])]
    if config.target_column not in df_cols:
        raise HTTPException(status_code=400, detail=f"Target column '{config.target_column}' not found in dataset")

    catalog = _model_catalog_payload()

    if config.cv_folds and config.cv_folds > MAX_CV_FOLDS:
        raise HTTPException(
            status_code=400,
            detail=f"cv_folds cannot exceed {MAX_CV_FOLDS} for demo tier",
        )

    if config.models_to_train:
        if len(config.models_to_train) > MAX_MODELS_PER_RUN:
            raise HTTPException(
                status_code=400,
                detail=f"Select at most {MAX_MODELS_PER_RUN} models per run",
            )

        invalid = sorted(set(config.models_to_train) - set(catalog["all"]))
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported models requested: {', '.join(invalid)}",
            )

        if config.task_type in {"classification", "regression"}:
            allowed = set(catalog[config.task_type])
            incompatible = [m for m in config.models_to_train if m not in allowed]
            if incompatible:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Selected models incompatible with task_type '{config.task_type}': "
                        + ", ".join(incompatible)
                    ),
                )

    job_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    try:
        baseline_df = _read_dataset_file(dataset.file_path)
        if len(baseline_df) > MAX_TRAINING_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset too large for demo tier. Max {MAX_TRAINING_ROWS} rows.",
            )
        drift_baseline = build_drift_baseline_snapshot(baseline_df, target_column=config.target_column)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Could not build drift baseline snapshot for dataset=%s reason=%s", dataset.id, exc)
        drift_baseline = None

    job = TrainingJob(
        job_id=job_id,
        dataset_id=config.dataset_id,
        target_column=config.target_column,
        task_type=config.task_type,
        status="local_pending" if config.execution_mode == "local" else "pending",
        progress=0,
        config={
            "target_column": config.target_column,
            "dataset_path": dataset.file_path,
            "task_type": config.task_type,
            "execution_mode": config.execution_mode,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "models_to_train": config.models_to_train,
            "enable_tuning": config.enable_tuning,
            "tuning_trials": config.tuning_trials,
            "tuning_time_budget_sec": config.tuning_time_budget_sec,
            "model_hyperparams": config.model_hyperparams,
            "experiment_run_id": run_id,
            "drift_baseline": drift_baseline,
        },
    )

    run = ExperimentRun(
        run_id=run_id,
        job_id=job_id,
        dataset_id=config.dataset_id,
        target_column=config.target_column,
        task_type=config.task_type,
        status="local_pending" if config.execution_mode == "local" else "pending",
        config={
            "execution_mode": config.execution_mode,
            "task_type": config.task_type,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "models_to_train": config.models_to_train,
            "enable_tuning": config.enable_tuning,
            "tuning_trials": config.tuning_trials,
            "tuning_time_budget_sec": config.tuning_time_budget_sec,
            "model_hyperparams": config.model_hyperparams,
            "dataset_path": dataset.file_path,
            "drift_baseline": drift_baseline,
        },
    )
    db.add(job)
    db.add(run)
    db.commit()
    db.refresh(job)

    if config.execution_mode == "local":
        output_dir = os.path.abspath(os.path.join("./backend/storage/models", "local_runs", job_id))
        local_spec = {
            "job_id": job_id,
            "run_id": run_id,
            "dataset_id": dataset.id,
            "dataset_name": dataset.name,
            "dataset_path": os.path.abspath(dataset.file_path),
            "target_column": config.target_column,
            "task_type": config.task_type,
            "model_config": {
                "models_to_train": config.models_to_train or [],
                "model_hyperparams": config.model_hyperparams or {},
            },
            "hyperparameters": {
                "test_size": config.test_size,
                "cv_folds": config.cv_folds,
                "enable_tuning": config.enable_tuning,
                "tuning_trials": config.tuning_trials,
                "tuning_time_budget_sec": config.tuning_time_budget_sec,
            },
            "output_dir": output_dir,
        }
        job.config = {
            **(job.config or {}),
            "local_job_spec": local_spec,
        }
        db.commit()
        db.refresh(job)
        logger.info("Local training job %s prepared (no server training executed)", job_id)
        return {
            **TrainingJobResponse.model_validate(job).model_dump(),
            "execution_mode": "local",
            "local_job_spec": local_spec,
            "message": "Run this job with the local agent on your machine, then sync results.",
        }

    from app.core.config import settings
    background_tasks.add_task(
        run_training,
        job_id=job_id,
        run_id=run_id,
        dataset_path=dataset.file_path,
        config={
            "target_column": config.target_column,
            "task_type": config.task_type,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "models_to_train": config.models_to_train,
            "enable_tuning": config.enable_tuning,
            "tuning_trials": config.tuning_trials,
            "tuning_time_budget_sec": config.tuning_time_budget_sec,
            "model_hyperparams": config.model_hyperparams,
        },
        db_url=settings.DATABASE_URL,
    )

    logger.info(f"Training job {job_id} queued")
    return {
        **TrainingJobResponse.model_validate(job).model_dump(),
        "execution_mode": "remote",
        "local_job_spec": None,
        "message": "Remote training queued on server.",
    }


@router.get("/training-status/{job_id}", response_model=TrainingJobResponse)
def get_training_status(job_id: str, db: Session = Depends(get_db)):
    """Get training job status and progress."""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job


@router.get("/training-jobs")
def list_training_jobs(db: Session = Depends(get_db)):
    """List all training jobs."""
    jobs = db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).all()
    return jobs


@router.get("/training/local-job-spec/{job_id}")
def get_local_job_spec(job_id: str, db: Session = Depends(get_db)):
    """Return local job spec for a local execution-mode job."""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    execution_mode = (job.config or {}).get("execution_mode", "remote")
    if execution_mode != "local":
        raise HTTPException(status_code=400, detail="Job is not in local execution mode")

    local_spec = (job.config or {}).get("local_job_spec")
    if not local_spec:
        raise HTTPException(status_code=404, detail="Local job spec not found")

    return local_spec


@router.post("/training/local-sync")
def sync_local_training_results(
    payload: LocalTrainingSyncPayload,
    db: Session = Depends(get_db),
):
    """Sync locally trained model metrics/artifacts back to backend."""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == payload.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    run = db.query(ExperimentRun).filter(ExperimentRun.job_id == payload.job_id).first()
    execution_mode = (job.config or {}).get("execution_mode", "remote")
    if execution_mode != "local":
        raise HTTPException(status_code=400, detail="Job is not configured for local execution")

    # Replace previous local model sync records for idempotent retries.
    db.query(TrainedModel).filter(TrainedModel.job_id == payload.job_id).delete()

    for m in payload.models:
        metrics = {
            **(m.metrics or {}),
            "storage_origin": "local",
            "execution_mode": "local",
            "artifact_ref": m.artifact_ref,
        }
        trained_model = TrainedModel(
            job_id=payload.job_id,
            model_name=m.model_name,
            model_type=m.model_type,
            task_type=m.task_type or payload.task_type or job.task_type,
            file_path=(m.artifact_ref or {}).get("path") if m.artifact_ref else None,
            accuracy=metrics.get("accuracy"),
            f1_score=metrics.get("f1_score") or metrics.get("f1"),
            roc_auc=metrics.get("roc_auc"),
            rmse=metrics.get("rmse"),
            mae=metrics.get("mae"),
            r2_score=metrics.get("r2") or metrics.get("r2_score"),
            metrics=metrics,
            feature_importance=m.feature_importance,
            confusion_matrix=m.confusion_matrix,
            roc_curve_data=m.roc_curve_data,
            cv_scores=m.cv_scores,
            training_time=m.training_time,
        )
        db.add(trained_model)

    if payload.task_type:
        job.task_type = payload.task_type
        if run:
            run.task_type = payload.task_type

    job.status = payload.status
    job.progress = 100 if payload.status == "completed" else job.progress
    job.error_message = payload.error_message
    job.completed_at = datetime.utcnow() if payload.status in {"completed", "failed"} else job.completed_at

    job.config = {
        **(job.config or {}),
        "local_sync": {
            "execution_env": payload.execution_env,
            "logs": payload.logs,
            "synced_at": datetime.utcnow().isoformat(),
        },
    }

    if run:
        run.status = payload.status
        run.error_message = payload.error_message
        run.completed_at = datetime.utcnow() if payload.status in {"completed", "failed"} else run.completed_at
        if payload.models:
            primary_metric = "accuracy" if (job.task_type or "") == "classification" else "r2"
            best = max(
                payload.models,
                key=lambda item: float((item.metrics or {}).get(primary_metric, (item.metrics or {}).get("r2_score", -1e9)) or -1e9),
            )
            best_score = (best.metrics or {}).get(primary_metric, (best.metrics or {}).get("r2_score"))
            run.best_model_name = best.model_name
            run.best_score = float(best_score) if best_score is not None else None
            run.summary_metrics = {
                "primary_metric": primary_metric,
                "best_model": best.model_name,
                "best_score": best_score,
                "models_trained": len(payload.models),
                "execution_mode": "local",
            }

    db.commit()
    return {
        "message": "Local training results synced successfully",
        "job_id": payload.job_id,
        "models_received": len(payload.models),
    }


@router.get("/training/model-catalog")
def training_model_catalog():
    """List trainable models by task type for UI model selection."""
    return _model_catalog_payload()


@router.get("/training/local-agent/download")
def download_local_agent():
    """Download the local training agent script."""
    agent_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "local_agent.py")
    )
    if not os.path.exists(agent_path):
        raise HTTPException(status_code=404, detail="local_agent.py not found")
    return FileResponse(
        path=agent_path,
        filename="local_agent.py",
        media_type="text/x-python",
    )


@router.get("/experiments", response_model=list[ExperimentRunResponse])
def list_experiments(db: Session = Depends(get_db), status: Optional[str] = None):
    """List experiment runs with optional status filtering."""
    query = db.query(ExperimentRun)
    if status:
        query = query.filter(ExperimentRun.status == status)
    return query.order_by(ExperimentRun.started_at.desc()).all()


@router.get("/experiments/{run_id}", response_model=ExperimentRunResponse)
def get_experiment(run_id: str, db: Session = Depends(get_db)):
    """Get a single experiment run by run_id."""
    run = db.query(ExperimentRun).filter(ExperimentRun.run_id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Experiment run not found")
    return run
