"""
Axiom Cloud AI - Predictions API Router
Run inference on deployed models.
"""
import joblib
import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.core.database import get_db
from app.models.db_models import TrainedModel
from app.schemas.schemas import PredictionRequest, PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory model cache
_model_cache: Dict[int, Any] = {}


def load_model(model_id: int, file_path: str) -> dict:
    """Load and cache a model from disk."""
    if model_id not in _model_cache:
        _model_cache[model_id] = joblib.load(file_path)
        logger.info(f"Model {model_id} loaded into cache")
    return _model_cache[model_id]


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    """Run inference using a deployed model."""
    model_record = db.query(TrainedModel).filter(TrainedModel.id == request.model_id).first()
    if not model_record:
        raise HTTPException(status_code=404, detail="Model not found")

    if not model_record.file_path:
        raise HTTPException(status_code=400, detail="Model file not available")

    try:
        model_data = load_model(request.model_id, model_record.file_path)
        pipeline = model_data["pipeline"]
        feature_names = model_data["feature_names"]
        label_encoder = model_data.get("label_encoder")
        task_type = model_data.get("task_type")

        # Build input dataframe
        input_data = {}
        for feat in feature_names:
            if feat in request.features:
                input_data[feat] = [request.features[feat]]
            else:
                input_data[feat] = [None]  # Will be imputed

        df_input = pd.DataFrame(input_data)

        prediction_raw = pipeline.predict(df_input)
        prediction = prediction_raw[0]

        # Decode label for classification
        probability = None
        confidence = None

        if task_type == "classification":
            if label_encoder:
                prediction = label_encoder.inverse_transform([int(prediction)])[0]

            if hasattr(pipeline, "predict_proba"):
                proba = pipeline.predict_proba(df_input)[0]
                classes = label_encoder.classes_ if label_encoder else [str(i) for i in range(len(proba))]
                probability = {str(cls): float(p) for cls, p in zip(classes, proba)}
                confidence = float(max(proba))
        else:
            prediction = float(prediction)

        return PredictionResponse(
            model_id=request.model_id,
            model_name=model_record.model_name,
            prediction=prediction,
            probability=probability,
            confidence=confidence,
        )

    except Exception as e:
        logger.exception(f"Prediction error for model {request.model_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/deployed-models")
def list_deployed_models(db: Session = Depends(get_db)):
    """List all currently deployed models."""
    models = db.query(TrainedModel).filter(TrainedModel.is_deployed == True).all()
    return [
        {
            "id": m.id,
            "model_name": m.model_name,
            "model_type": m.model_type,
            "task_type": m.task_type,
            "job_id": m.job_id,
        }
        for m in models
    ]
