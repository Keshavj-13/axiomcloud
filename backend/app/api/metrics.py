"""
Axiom Cloud AI - Metrics API Router
Returns comparison metrics and visualization data.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import os
import pandas as pd

from app.core.database import get_db
from app.models.db_models import TrainedModel, TrainingJob, Dataset
from app.schemas.schemas import MetricsResponseV2
from app.ml.metrics_payload import build_metrics_payload
from app.ml.dataset_profiling import build_dataset_profile

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/metrics/{job_id}", response_model=MetricsResponseV2)
def get_metrics(job_id: str, db: Session = Depends(get_db)):
    """Get all model metrics for a training job — for leaderboard & charts."""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    models = db.query(TrainedModel).filter(TrainedModel.job_id == job_id).all()
    task_type = job.task_type or (models[0].task_type if len(models) > 0 else "unknown")

    dataset_profile = None
    dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
    if dataset and dataset.file_path and os.path.exists(dataset.file_path):
        try:
            df = pd.read_csv(dataset.file_path) if dataset.file_path.endswith(".csv") else pd.read_excel(dataset.file_path)
            dataset_profile = build_dataset_profile(
                df,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                target_column=job.target_column,
            )
        except Exception as exc:
            logger.warning("Could not build dataset profile for metrics payload | job=%s reason=%s", job_id, exc)

    return build_metrics_payload(
        job_id=job_id,
        task_type=task_type,
        status=job.status,
        models=models,
        dataset_profile=dataset_profile,
    )


@router.get("/dashboard/summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    """Get overall platform statistics for the dashboard."""
    total_datasets = db.query(TrainingJob).count()
    total_models = db.query(TrainedModel).count()
    deployed_models = db.query(TrainedModel).filter(TrainedModel.is_deployed == True).count()
    completed_jobs = db.query(TrainingJob).filter(TrainingJob.status == "completed").count()
    failed_jobs = db.query(TrainingJob).filter(TrainingJob.status == "failed").count()

    return {
        "total_datasets": total_datasets,
        "total_models": total_models,
        "deployed_models": deployed_models,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
    }
