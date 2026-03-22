"""
Axiom Cloud AI - Pydantic Schemas for API validation
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime


# ─── Dataset Schemas ──────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: List[Any]
    stats: Optional[Dict[str, Any]] = None


class DatasetBase(BaseModel):
    name: str
    target_column: Optional[str] = None
    task_type: Optional[str] = None


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    file_size: Optional[int]
    num_rows: Optional[int]
    num_columns: Optional[int]
    columns_info: Optional[List[Dict[str, Any]]]
    is_example: bool
    preview_data: Optional[List[Dict[str, Any]]]
    created_at: datetime


# ─── Training Schemas ─────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    dataset_id: int
    target_column: str
    task_type: Optional[str] = None  # auto-detect if None
    models_to_train: Optional[List[str]] = None  # train all if None
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    cv_folds: int = Field(default=5, ge=2, le=10)


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: str
    dataset_id: int
    task_type: Optional[str]
    target_column: str
    status: str
    progress: int
    error_message: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]


# ─── Model Schemas ────────────────────────────────────────────────────────────

class ModelResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    job_id: str
    model_name: str
    model_type: str
    task_type: str
    is_deployed: bool
    accuracy: Optional[float]
    f1_score: Optional[float]
    roc_auc: Optional[float]
    rmse: Optional[float]
    mae: Optional[float]
    r2_score: Optional[float]
    metrics: Optional[Dict[str, Any]]
    feature_importance: Optional[Dict[str, float]]
    confusion_matrix: Optional[List[List[int]]]
    roc_curve_data: Optional[Dict[str, List[float]]]
    cv_scores: Optional[List[float]]
    training_time: Optional[float]
    created_at: datetime


# ─── Prediction Schemas ───────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: int
    features: Dict[str, Any]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: int
    model_name: str
    prediction: Any
    probability: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None


# ─── Metrics Schemas ──────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    job_id: str
    models: List[ModelResponse]
    best_model: Optional[ModelResponse]
    task_type: str
