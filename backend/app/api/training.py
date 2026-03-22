"""
Axiom Cloud AI - Training API Router
Launches AutoML training jobs (sync for simplicity, async with Celery in prod).
"""
import uuid
import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from app.core.database import get_db
from app.models.db_models import Dataset, TrainingJob, TrainedModel, ExperimentRun
from app.schemas.schemas import TrainingConfig, TrainingJobResponse, ExperimentRunResponse
from app.ml.pipeline import AutoMLPipeline

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory progress tracking (use Redis in production)
_job_progress: dict = {}


def run_training(job_id: str, run_id: str, dataset_path: str, config: dict, db_url: str):
    """Background training function."""
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
        df = pd.read_csv(dataset_path)

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
        )

        # Update job task_type
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        job.task_type = results["task_type"]

        # Save model results
        for model_result in results["models"]:
            if "error" in model_result:
                continue

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
                metrics=model_result.get("metrics"),
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


@router.post("/train-model", response_model=TrainingJobResponse)
async def train_model(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Launch an AutoML training job."""
    # Validate dataset
    dataset = db.query(Dataset).filter(Dataset.id == config.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate target column
    df_cols = [c["name"] for c in (dataset.columns_info or [])]
    if config.target_column not in df_cols:
        raise HTTPException(status_code=400, detail=f"Target column '{config.target_column}' not found in dataset")

    job_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())

    job = TrainingJob(
        job_id=job_id,
        dataset_id=config.dataset_id,
        target_column=config.target_column,
        task_type=config.task_type,
        status="pending",
        progress=0,
        config={
            "target_column": config.target_column,
            "dataset_path": dataset.file_path,
            "task_type": config.task_type,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "models_to_train": config.models_to_train,
            "enable_tuning": config.enable_tuning,
            "tuning_trials": config.tuning_trials,
            "tuning_time_budget_sec": config.tuning_time_budget_sec,
            "experiment_run_id": run_id,
        },
    )

    run = ExperimentRun(
        run_id=run_id,
        job_id=job_id,
        dataset_id=config.dataset_id,
        target_column=config.target_column,
        task_type=config.task_type,
        status="pending",
        config={
            "task_type": config.task_type,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
            "models_to_train": config.models_to_train,
            "enable_tuning": config.enable_tuning,
            "tuning_trials": config.tuning_trials,
            "tuning_time_budget_sec": config.tuning_time_budget_sec,
            "dataset_path": dataset.file_path,
        },
    )
    db.add(job)
    db.add(run)
    db.commit()
    db.refresh(job)

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
        },
        db_url=settings.DATABASE_URL,
    )

    logger.info(f"Training job {job_id} queued")
    return job


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
