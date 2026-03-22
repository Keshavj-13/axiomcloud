"""
Axiom Cloud AI - Models API Router
List, retrieve, deploy, and download trained models.
"""
import os
import logging
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
import numpy as np

from app.core.database import get_db
from app.models.db_models import TrainedModel, TrainingJob, Dataset
from app.schemas.schemas import (
    ModelResponse,
    APIErrorResponse,
    ShapSummaryResponse,
    LimeSummaryResponse,
    LimeExplainRequest,
)

import pandas as pd
from app.ml.explainability import explainability_service, ExplainabilityError

router = APIRouter()
logger = logging.getLogger(__name__)


def _load_dataset_frame(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


@router.get("/models", response_model=List[ModelResponse])
def list_models(
    job_id: str = None,
    db: Session = Depends(get_db)
):
    """List trained models, optionally filtered by job."""
    query = db.query(TrainedModel)
    if job_id:
        query = query.filter(TrainedModel.job_id == job_id)
    return query.order_by(TrainedModel.created_at.desc()).all()


@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get a specific model by ID."""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


# --- Explainability Endpoints ---
@router.get(
    "/models/{model_id}/shap",
    response_model=ShapSummaryResponse,
    responses={400: {"model": APIErrorResponse}, 404: {"model": APIErrorResponse}, 500: {"model": APIErrorResponse}},
)
def get_model_shap(
    model_id: int,
    sample_index: int = Query(default=0, ge=0),
    nsamples: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Return chart-ready SHAP global/local explanations for a selected sample."""
    model_obj = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model_obj or not model_obj.file_path or not os.path.exists(model_obj.file_path):
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "MODEL_NOT_FOUND", "message": "Model not found or file missing"}},
        )

    job = db.query(TrainingJob).filter(TrainingJob.job_id == model_obj.job_id).first()
    if not job or not job.config or not job.config.get("dataset_path"):
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "TRAINING_DATA_NOT_FOUND", "message": "Training data not found for SHAP explanation"}},
        )

    try:
        dataset_path = job.config.get("dataset_path")
        return explainability_service.build_shap(
            model=model_obj,
            dataset_path=dataset_path,
            target_column=job.target_column,
            sample_index=sample_index,
            nsamples=nsamples,
        )
    except ExplainabilityError as exc:
        status = 400
        if exc.code in {"MODEL_FILE_MISSING", "DATASET_FILE_MISSING", "TARGET_COLUMN_MISSING"}:
            status = 404
        logger.warning("SHAP request failed | model_id=%s code=%s msg=%s", model_id, exc.code, exc.message)
        raise HTTPException(
            status_code=status,
            detail={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        ) from exc
    except Exception:
        logger.exception("SHAP explanation failed unexpectedly")
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "SHAP_INTERNAL_ERROR", "message": "SHAP explanation failed"}},
        )


@router.post(
    "/models/{model_id}/lime",
    response_model=LimeSummaryResponse,
    responses={400: {"model": APIErrorResponse}, 404: {"model": APIErrorResponse}, 500: {"model": APIErrorResponse}},
)
def get_model_lime(
    model_id: int,
    payload: LimeExplainRequest = Body(default_factory=LimeExplainRequest),
    db: Session = Depends(get_db),
):
    """Return chart-ready LIME local explanations for a selected sample or custom input."""
    model_obj = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model_obj or not model_obj.file_path or not os.path.exists(model_obj.file_path):
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "MODEL_NOT_FOUND", "message": "Model not found or file missing"}},
        )

    job = db.query(TrainingJob).filter(TrainingJob.job_id == model_obj.job_id).first()
    if not job or not job.config or not job.config.get("dataset_path"):
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "TRAINING_DATA_NOT_FOUND", "message": "Training data not found for LIME explanation"}},
        )

    try:
        dataset_path = job.config.get("dataset_path")
        return explainability_service.build_lime(
            model=model_obj,
            dataset_path=dataset_path,
            target_column=job.target_column,
            sample_index=payload.sample_index,
            num_features=payload.num_features,
            custom_input=payload.custom_input,
        )
    except ExplainabilityError as exc:
        status = 400
        if exc.code in {"MODEL_FILE_MISSING", "DATASET_FILE_MISSING", "TARGET_COLUMN_MISSING"}:
            status = 404
        logger.warning("LIME request failed | model_id=%s code=%s msg=%s", model_id, exc.code, exc.message)
        raise HTTPException(
            status_code=status,
            detail={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        ) from exc
    except Exception:
        logger.exception("LIME explanation failed unexpectedly")
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "LIME_INTERNAL_ERROR", "message": "LIME explanation failed"}},
        )


@router.get(
    "/models/{model_id}/lime",
    response_model=LimeSummaryResponse,
    responses={400: {"model": APIErrorResponse}, 404: {"model": APIErrorResponse}, 500: {"model": APIErrorResponse}},
)
def get_model_lime_compat(
    model_id: int,
    sample_index: int = Query(default=0, ge=0),
    num_features: int = Query(default=12, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Backward-compatible GET endpoint for LIME explanations."""
    payload = LimeExplainRequest(sample_index=sample_index, num_features=num_features, custom_input=None)
    return get_model_lime(model_id=model_id, payload=payload, db=db)


@router.get("/models/{model_id}/monitoring")
def get_model_monitoring(
    model_id: int,
    compare_dataset_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Return model monitoring snapshot and optional drift report."""
    model_obj = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model_obj:
        raise HTTPException(status_code=404, detail="Model not found")

    job = db.query(TrainingJob).filter(TrainingJob.job_id == model_obj.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found for model")

    cv_scores = model_obj.cv_scores or []
    cv_mean = float(np.mean(cv_scores)) if cv_scores else None
    cv_std = float(np.std(cv_scores)) if cv_scores else None

    monitoring_snapshot = {
        "model_id": model_obj.id,
        "model_name": model_obj.model_name,
        "task_type": model_obj.task_type,
        "is_deployed": model_obj.is_deployed,
        "training_time": model_obj.training_time,
        "primary_metric": (
            model_obj.accuracy
            if model_obj.task_type == "classification"
            else model_obj.r2_score
        ),
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "stability": (
            "stable" if (cv_std is not None and cv_std < 0.03)
            else "moderate" if (cv_std is not None and cv_std < 0.08)
            else "unstable" if cv_std is not None
            else "unknown"
        ),
    }

    drift_report = None
    if compare_dataset_id is not None:
        train_dataset = db.query(Dataset).filter(Dataset.id == job.dataset_id).first()
        compare_dataset = db.query(Dataset).filter(Dataset.id == compare_dataset_id).first()
        if not train_dataset or not compare_dataset:
            raise HTTPException(status_code=404, detail="Compare dataset or training dataset not found")
        if not os.path.exists(train_dataset.file_path) or not os.path.exists(compare_dataset.file_path):
            raise HTTPException(status_code=404, detail="Dataset file not found on disk")

        train_df = _load_dataset_frame(train_dataset.file_path)
        current_df = _load_dataset_frame(compare_dataset.file_path)

        # Exclude target column if present
        target_col = job.target_column
        train_X = train_df.drop(columns=[target_col], errors="ignore")
        current_X = current_df.drop(columns=[target_col], errors="ignore")

        shared_numeric = [
            c for c in train_X.select_dtypes(include=["int64", "float64"]).columns
            if c in current_X.columns
        ]

        per_feature = []
        for c in shared_numeric[:30]:
            train_col = train_X[c].dropna()
            curr_col = current_X[c].dropna()
            if train_col.empty or curr_col.empty:
                continue

            train_mean = float(train_col.mean())
            curr_mean = float(curr_col.mean())
            train_std = float(train_col.std()) if float(train_col.std()) > 1e-9 else 1.0
            z_drift = abs(curr_mean - train_mean) / train_std

            per_feature.append({
                "feature": c,
                "train_mean": train_mean,
                "current_mean": curr_mean,
                "z_drift": round(float(z_drift), 4),
            })

        avg_drift = float(np.mean([f["z_drift"] for f in per_feature])) if per_feature else 0.0
        drift_level = "low" if avg_drift < 0.5 else "medium" if avg_drift < 1.0 else "high"

        drift_report = {
            "train_dataset_id": train_dataset.id,
            "compare_dataset_id": compare_dataset.id,
            "features_compared": len(per_feature),
            "avg_z_drift": round(avg_drift, 4),
            "drift_level": drift_level,
            "top_drift_features": sorted(per_feature, key=lambda x: x["z_drift"], reverse=True)[:10],
        }

    return {
        "monitoring": monitoring_snapshot,
        "drift": drift_report,
    }


@router.post("/models/{model_id}/deploy")
def deploy_model(model_id: int, db: Session = Depends(get_db)):
    """Mark a model as deployed (active for predictions)."""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Un-deploy other models from same job
    db.query(TrainedModel).filter(
        TrainedModel.job_id == model.job_id,
        TrainedModel.id != model_id
    ).update({"is_deployed": False})

    model.is_deployed = True
    db.commit()
    logger.info(f"🚀 Model {model_id} ({model.model_name}) deployed")
    return {"message": f"Model '{model.model_name}' deployed successfully", "model_id": model_id}


@router.post("/models/{model_id}/undeploy")
def undeploy_model(model_id: int, db: Session = Depends(get_db)):
    """Undeploy a model."""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    model.is_deployed = False
    db.commit()
    return {"message": "Model undeployed"}


@router.get("/models/{model_id}/download")
def download_model(model_id: int, db: Session = Depends(get_db)):
    """Download a trained model file (joblib)."""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if not model.file_path or not os.path.exists(model.file_path):
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    filename = f"{model.model_name.replace(' ', '_')}_{model_id}.joblib"
    return FileResponse(
        path=model.file_path,
        media_type="application/octet-stream",
        filename=filename,
    )


@router.delete("/models/{model_id}")
def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Delete a trained model."""
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    if model.file_path and os.path.exists(model.file_path):
        os.remove(model.file_path)

    db.delete(model)
    db.commit()
    return {"message": "Model deleted successfully"}
