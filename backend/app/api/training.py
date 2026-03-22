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
from app.models.db_models import Dataset, TrainingJob, TrainedModel
from app.schemas.schemas import TrainingConfig, TrainingJobResponse
from app.ml.pipeline import AutoMLPipeline

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory progress tracking (use Redis in production)
_job_progress: dict = {}


def run_training(job_id: str, dataset_path: str, config: dict, db_url: str):
    """Background training function."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        # Update job status
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            return
        job.status = "running"
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

        job.status = "completed"
        job.completed_at = datetime.utcnow()
        job.progress = 100
        db.commit()
        logger.info(f"✅ Training job {job_id} completed")

    except Exception as e:
        logger.exception(f"❌ Training job {job_id} failed: {e}")
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(e)
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
        },
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    from app.core.config import settings
    background_tasks.add_task(
        run_training,
        job_id=job_id,
        dataset_path=dataset.file_path,
        config={
            "target_column": config.target_column,
            "task_type": config.task_type,
            "test_size": config.test_size,
            "cv_folds": config.cv_folds,
        },
        db_url=settings.DATABASE_URL,
    )

    logger.info(f"🚀 Training job {job_id} queued")
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
