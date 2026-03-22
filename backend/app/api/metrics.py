"""
SigmaCloud AI - Metrics API Router
Returns comparison metrics and visualization data.
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.models.db_models import TrainedModel, TrainingJob

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/metrics/{job_id}")
def get_metrics(job_id: str, db: Session = Depends(get_db)):
    """Get all model metrics for a training job — for leaderboard & charts."""
    job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    models = db.query(TrainedModel).filter(TrainedModel.job_id == job_id).all()
    if not models:
        return {"job_id": job_id, "status": job.status, "models": []}

    task_type = job.task_type or (models[0].task_type if models else "unknown")

    model_data = []
    for m in models:
        data = {
            "id": m.id,
            "model_name": m.model_name,
            "model_type": m.model_type,
            "task_type": m.task_type,
            "training_time": m.training_time,
            "is_deployed": m.is_deployed,
            "cv_scores": m.cv_scores,
            "metrics": m.metrics,
        }
        if task_type == "classification":
            data.update({
                "accuracy": m.accuracy,
                "f1_score": m.f1_score,
                "roc_auc": m.roc_auc,
                "precision": (m.metrics or {}).get("precision") if m.metrics else None,
                "recall": (m.metrics or {}).get("recall") if m.metrics else None,
                "balanced_accuracy": (m.metrics or {}).get("balanced_accuracy") if m.metrics else None,
                "confusion_matrix": m.confusion_matrix,
                "roc_curve_data": m.roc_curve_data,
            })
        else:
            data.update({
                "rmse": m.rmse,
                "mae": m.mae,
                "r2_score": m.r2_score,
                "mape": (m.metrics or {}).get("mape") if m.metrics else None,
                "explained_variance": (m.metrics or {}).get("explained_variance") if m.metrics else None,
                "median_ae": (m.metrics or {}).get("median_ae") if m.metrics else None,
            })

        data["feature_importance"] = m.feature_importance
        model_data.append(data)

    # Sort by best metric
    if task_type == "classification":
        model_data.sort(key=lambda x: x.get("accuracy") or 0, reverse=True)
        best_model = model_data[0] if model_data else None
    else:
        model_data.sort(key=lambda x: x.get("r2_score") or -999, reverse=True)
        best_model = model_data[0] if model_data else None

    return {
        "job_id": job_id,
        "task_type": task_type,
        "status": job.status,
        "models": model_data,
        "best_model": best_model,
        "leaderboard": [
            {
                "rank": i + 1,
                "model_name": m["model_name"],
                "primary_metric": m.get("accuracy") or m.get("r2_score"),
                "metric_name": "accuracy" if task_type == "classification" else "r2_score",
                "training_time": m.get("training_time"),
            }
            for i, m in enumerate(model_data)
        ],
    }


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
