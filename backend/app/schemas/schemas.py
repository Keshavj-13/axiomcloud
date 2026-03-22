"""
Axiom Cloud AI - Pydantic Schemas for API validation
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
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
    enable_tuning: bool = False
    tuning_trials: int = Field(default=12, ge=3, le=200)
    tuning_time_budget_sec: int = Field(default=180, ge=30, le=3600)


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


class ExperimentRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: str
    job_id: str
    dataset_id: int
    target_column: str
    task_type: Optional[str]
    status: str
    config: Optional[Dict[str, Any]]
    summary_metrics: Optional[Dict[str, Any]]
    best_model_name: Optional[str]
    best_score: Optional[float]
    error_message: Optional[str]
    started_at: datetime
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


# ─── Structured Error Schemas ────────────────────────────────────────────────

class APIError(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class APIErrorResponse(BaseModel):
    error: APIError


# ─── Explainability Schemas ─────────────────────────────────────────────────

class ExplainabilityMetadata(BaseModel):
    model_id: int
    model_name: str
    model_type: str
    task_type: Literal["classification", "regression", "unknown"]
    job_id: str
    artifact_hash: str
    dataset_hash: str
    cache_key: str
    used_fallback: bool = False
    explainer: str


class ExplainabilitySamplePrediction(BaseModel):
    sample_index: int
    prediction: float | int | str
    prediction_label: Optional[str] = None
    confidence: Optional[float] = None
    class_probabilities: Optional[Dict[str, float]] = None


class FeatureContribution(BaseModel):
    feature: str
    value: float
    abs_value: float
    rank: int


class ShapGlobalFeatureImportance(BaseModel):
    feature: str
    mean_abs_contribution: float
    rank: int


class ShapSummaryResponse(BaseModel):
    metadata: ExplainabilityMetadata
    feature_names: List[str]
    expected_value: float | List[float] | None
    global_importance: List[ShapGlobalFeatureImportance]
    local_contributions: List[FeatureContribution]
    sample_prediction: ExplainabilitySamplePrediction
    chart: Dict[str, Any]
    model_metadata: Dict[str, Any]


class ShapExplainRequest(BaseModel):
    sample_index: int = Field(default=0, ge=0)
    nsamples: int = Field(default=200, ge=1, le=1000)


class LimeContribution(BaseModel):
    feature: str
    weight: float
    abs_weight: float
    direction: Literal["positive", "negative"]
    rank: int


class LimeSummaryResponse(BaseModel):
    metadata: ExplainabilityMetadata
    feature_names: List[str]
    sample_prediction: ExplainabilitySamplePrediction
    weights: List[LimeContribution]
    top_positive: List[LimeContribution]
    top_negative: List[LimeContribution]
    class_context: Dict[str, Any]
    chart: Dict[str, Any]


class LimeExplainRequest(BaseModel):
    sample_index: int = Field(default=0, ge=0)
    num_features: int = Field(default=12, ge=1, le=100)
    custom_input: Optional[Dict[str, Any]] = None


# ─── Metrics Payload Schemas ────────────────────────────────────────────────

class ChartSeries(BaseModel):
    labels: List[str]
    series: Dict[str, List[float | int | None]]


class ConfusionMatrixChart(BaseModel):
    labels: List[str]
    matrix: List[List[int]]
    normalized_matrix: Optional[List[List[float]]] = None


class RocCurveChart(BaseModel):
    fpr: List[float]
    tpr: List[float]
    thresholds: Optional[List[float]] = None


class ModelMetricsExtended(BaseModel):
    id: int
    model_name: str
    model_type: str
    task_type: str
    training_time: Optional[float] = None
    is_deployed: bool
    metrics: Dict[str, Any] = Field(default_factory=dict)
    cv_scores: Optional[List[float]] = None
    per_fold: Optional[List[Dict[str, Any]]] = None
    per_class: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None
    confusion_matrix_chart: Optional[ConfusionMatrixChart] = None
    roc_curve_chart: Optional[RocCurveChart] = None
    # Backward-compatible flattened metrics
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    mape: Optional[float] = None
    explained_variance: Optional[float] = None
    median_ae: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    roc_curve_data: Optional[Dict[str, Any]] = None


class DatasetProfileCompact(BaseModel):
    dataset_id: int
    dataset_name: str
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    missing_total: int
    missing_rate: float
    duplicate_rows: int
    columns: List[Dict[str, Any]]
    summary_cards: List[Dict[str, Any]]
    target_distribution: Optional[Dict[str, Any]] = None
    correlation_heatmap: Optional[Dict[str, Any]] = None
    typing_intelligence: Optional[Dict[str, Any]] = None
    leakage_risks: Optional[List[Dict[str, Any]]] = None
    drift_baseline: Optional[Dict[str, Any]] = None
    missing_by_column: List[Dict[str, Any]]
    histograms: List[Dict[str, Any]]


class MetricsResponseV2(BaseModel):
    job_id: str
    task_type: str
    status: str
    metric_catalog: List[str]
    models: List[ModelMetricsExtended]
    best_model_id: Optional[int] = None
    best_model: Optional[ModelMetricsExtended] = None
    leaderboard: List[Dict[str, Any]]
    chart_data: Dict[str, Any]
    dataset_profile: Optional[DatasetProfileCompact] = None
