"""
Axiom Cloud AI - Datasets API Router
Handles dataset upload, listing, and example datasets.
"""
import os
import uuid
import logging
import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, List

from app.core.database import get_db
from app.models.db_models import Dataset, TrainedModel
from app.schemas.schemas import DatasetResponse, DatasetProfileCompact
from app.ml.datasets import EXAMPLE_DATASETS
from app.ml.dataset_profiling import build_dataset_profile, build_drift_baseline_snapshot, build_eda_report
from app.ml.analytics_service import AnalyticsReportService, build_predictions_from_model_bundle
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)
analytics_service = AnalyticsReportService()

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def analyze_dataframe(df: pd.DataFrame) -> List[dict]:
    """Analyze dataframe columns and return metadata."""
    columns_info = []
    for col in df.columns:
        col_data = df[col]
        info = {
            "name": col,
            "dtype": str(col_data.dtype),
            "null_count": int(col_data.isnull().sum()),
            "unique_count": int(col_data.nunique()),
            "sample_values": col_data.dropna().head(5).tolist(),
        }

        if col_data.dtype in ["int64", "float64"]:
            info["stats"] = {
                "min": float(col_data.min()) if not col_data.empty else None,
                "max": float(col_data.max()) if not col_data.empty else None,
                "mean": float(col_data.mean()) if not col_data.empty else None,
                "std": float(col_data.std()) if not col_data.empty else None,
                "median": float(col_data.median()) if not col_data.empty else None,
            }

        columns_info.append(info)
    return columns_info


def _load_dataset_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def build_quality_report(df: pd.DataFrame) -> dict:
    """Build a lightweight data quality report for a dataset."""
    n_rows, n_cols = df.shape
    total_cells = max(n_rows * n_cols, 1)

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    missing_by_column = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if int(df[col].isna().sum()) > 0
    }

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    outliers_by_column = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = int(((s < lower) | (s > upper)).sum())
        if outliers > 0:
            outliers_by_column[col] = outliers

    quality_score = 100.0
    quality_score -= min(25.0, (missing_cells / total_cells) * 100)
    quality_score -= min(20.0, (duplicate_rows / max(n_rows, 1)) * 100)
    quality_score -= min(20.0, sum(outliers_by_column.values()) / max(n_rows, 1) * 100)
    quality_score = round(max(0.0, quality_score), 2)

    recommendations = []
    if missing_cells > 0:
        recommendations.append("Impute missing values (median for numeric, mode for categorical)")
    if duplicate_rows > 0:
        recommendations.append("Drop duplicate rows before training")
    if outliers_by_column:
        recommendations.append("Cap or transform outlier-heavy numeric columns")
    if not recommendations:
        recommendations.append("Dataset looks clean; proceed to training")

    return {
        "rows": n_rows,
        "columns": n_cols,
        "quality_score": quality_score,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "missing_by_column": missing_by_column,
        "outliers_by_column": outliers_by_column,
        "recommendations": recommendations,
    }


def build_clean_preview(df: pd.DataFrame, preview_rows: int = 10) -> dict:
    """Return cleaned preview and applied fixes (without mutating stored file)."""
    cleaned = df.copy()
    applied_fixes = []

    before_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    dropped = before_rows - len(cleaned)
    if dropped > 0:
        applied_fixes.append(f"Dropped {dropped} duplicate rows")

    numeric_cols = cleaned.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = cleaned.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    for col in numeric_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            applied_fixes.append(f"Filled missing numeric values in '{col}' with median")

    for col in cat_cols:
        if cleaned[col].isna().any():
            mode = cleaned[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else "unknown"
            cleaned[col] = cleaned[col].fillna(fill_val)
            applied_fixes.append(f"Filled missing categorical values in '{col}' with mode")

    # Clip numeric outliers using IQR
    for col in numeric_cols:
        s = cleaned[col].dropna()
        if len(s) < 5:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = cleaned[col].copy()
        cleaned[col] = cleaned[col].clip(lower, upper)
        clipped = int((before != cleaned[col]).sum())
        if clipped > 0:
            applied_fixes.append(f"Clipped {clipped} outlier values in '{col}'")

    preview = cleaned.head(preview_rows).fillna("").to_dict(orient="records")
    return {
        "preview_rows": preview_rows,
        "rows_after_cleaning": len(cleaned),
        "applied_fixes": applied_fixes,
        "clean_preview": preview,
    }


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Clean dataframe and return cleaned copy with applied fix messages."""
    cleaned = df.copy()
    applied_fixes = []

    before_rows = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    dropped = before_rows - len(cleaned)
    if dropped > 0:
        applied_fixes.append(f"Dropped {dropped} duplicate rows")

    numeric_cols = cleaned.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = cleaned.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    for col in numeric_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            applied_fixes.append(f"Filled missing numeric values in '{col}' with median")

    for col in cat_cols:
        if cleaned[col].isna().any():
            mode = cleaned[col].mode(dropna=True)
            fill_val = mode.iloc[0] if not mode.empty else "unknown"
            cleaned[col] = cleaned[col].fillna(fill_val)
            applied_fixes.append(f"Filled missing categorical values in '{col}' with mode")

    # Clip numeric outliers using IQR
    for col in numeric_cols:
        s = cleaned[col].dropna()
        if len(s) < 5:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before = cleaned[col].copy()
        cleaned[col] = cleaned[col].clip(lower, upper)
        clipped = int((before != cleaned[col]).sum())
        if clipped > 0:
            applied_fixes.append(f"Clipped {clipped} outlier values in '{col}'")

    return cleaned, applied_fixes


@router.post("/upload-dataset", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """Upload a CSV/Excel dataset for training."""
    # Validate file type
    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 5MB.")

    # Save file
    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{file.filename}"
    file_path = os.path.join(settings.DATASET_STORAGE_PATH, safe_name)

    with open(file_path, "wb") as f:
        f.write(content)

    # Parse
    try:
        df = _load_dataset_file(file_path)
    except Exception as e:
        logger.exception(f"Failed to parse uploaded dataset '{file.filename}': {e}")
        os.remove(file_path)
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    columns_info = analyze_dataframe(df)
    preview = df.head(10).fillna("").to_dict(orient="records")

    dataset = Dataset(
        name=name or file.filename,
        filename=file.filename,
        file_path=file_path,
        file_size=len(content),
        num_rows=len(df),
        num_columns=len(df.columns),
        columns_info=columns_info,
        target_column=target_column,
        is_example=False,
        preview_data=preview,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    logger.info(f"Dataset uploaded: {dataset.name} ({len(df)} rows)")
    return dataset


@router.post("/load-example/{dataset_key}", response_model=DatasetResponse)
async def load_example_dataset(
    dataset_key: str,
    db: Session = Depends(get_db),
):
    """Load a built-in example dataset."""
    if dataset_key not in EXAMPLE_DATASETS:
        raise HTTPException(status_code=404, detail=f"Example dataset '{dataset_key}' not found.")

    meta = EXAMPLE_DATASETS[dataset_key]

    # Check if already loaded
    existing = db.query(Dataset).filter(
        Dataset.is_example == True,
        Dataset.name == meta["name"]
    ).first()
    if existing:
        return existing

    df = meta["loader"]()
    file_path = os.path.join(settings.DATASET_STORAGE_PATH, meta["filename"])
    df.to_csv(file_path, index=False)

    columns_info = analyze_dataframe(df)
    preview = df.head(10).fillna("").to_dict(orient="records")

    dataset = Dataset(
        name=meta["name"],
        filename=meta["filename"],
        file_path=file_path,
        file_size=os.path.getsize(file_path),
        num_rows=len(df),
        num_columns=len(df.columns),
        columns_info=columns_info,
        target_column=meta["target"],
        task_type=meta["task_type"],
        is_example=True,
        preview_data=preview,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    logger.info(f"Example dataset loaded: {meta['name']}")
    return dataset


@router.get("/datasets", response_model=List[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)):
    """List all datasets."""
    return db.query(Dataset).order_by(Dataset.created_at.desc()).all()


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Get a specific dataset by ID."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


@router.get("/datasets/{dataset_id}/quality-report")
def dataset_quality_report(dataset_id: int, db: Session = Depends(get_db)):
    """Return data quality diagnostics and recommendations."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    try:
        df = _load_dataset_file(dataset.file_path)
    except Exception as e:
        logger.exception(f"Failed quality report for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=422, detail=f"Could not parse dataset: {e}")

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        **build_quality_report(df),
    }


@router.get("/datasets/{dataset_id}/clean-preview")
def dataset_clean_preview(dataset_id: int, preview_rows: int = 10, db: Session = Depends(get_db)):
    """Return a cleaned preview and applied fixes without changing source file."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    try:
        df = _load_dataset_file(dataset.file_path)
    except Exception as e:
        logger.exception(f"Failed clean preview for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=422, detail=f"Could not parse dataset: {e}")

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        **build_clean_preview(df, preview_rows=preview_rows),
    }


@router.post("/datasets/{dataset_id}/clean-and-save", response_model=DatasetResponse)
def dataset_clean_and_save(
    dataset_id: int,
    name_suffix: str = "cleaned",
    db: Session = Depends(get_db),
):
    """Create and store a cleaned copy of an existing dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    try:
        df = _load_dataset_file(dataset.file_path)
    except Exception as e:
        logger.exception(f"Failed clean-and-save for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=422, detail=f"Could not parse dataset: {e}")

    cleaned_df, applied_fixes = clean_dataframe(df)

    base_name = os.path.splitext(dataset.filename)[0]
    ext = os.path.splitext(dataset.filename)[1] or ".csv"
    clean_filename = f"{base_name}_{name_suffix}{ext if ext in ['.csv', '.xlsx', '.xls'] else '.csv'}"
    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{clean_filename}"
    clean_path = os.path.join(settings.DATASET_STORAGE_PATH, safe_name)

    if safe_name.endswith(".csv"):
        cleaned_df.to_csv(clean_path, index=False)
    else:
        cleaned_df.to_excel(clean_path, index=False)

    columns_info = analyze_dataframe(cleaned_df)
    preview = cleaned_df.head(10).fillna("").to_dict(orient="records")

    cleaned_dataset = Dataset(
        name=f"{dataset.name} ({name_suffix})",
        filename=clean_filename,
        file_path=clean_path,
        file_size=os.path.getsize(clean_path),
        num_rows=len(cleaned_df),
        num_columns=len(cleaned_df.columns),
        columns_info=columns_info,
        target_column=dataset.target_column,
        task_type=dataset.task_type,
        is_example=False,
        preview_data=preview,
    )
    db.add(cleaned_dataset)
    db.commit()
    db.refresh(cleaned_dataset)

    logger.info(
        "Created cleaned dataset copy %s from dataset %s with %d fixes",
        cleaned_dataset.id,
        dataset_id,
        len(applied_fixes),
    )
    return cleaned_dataset


@router.get("/example-datasets")
def list_example_datasets():
    """List available example datasets."""
    return [
        {
            "key": key,
            "name": meta["name"],
            "description": meta["description"],
            "target": meta["target"],
            "task_type": meta["task_type"],
        }
        for key, meta in EXAMPLE_DATASETS.items()
    ]


@router.get("/datasets/{dataset_id}/profile", response_model=DatasetProfileCompact)
def dataset_profile(dataset_id: int, db: Session = Depends(get_db)):
    """Return compact dataset profiling payload for interactive visual analytics."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    try:
        df = _load_dataset_file(dataset.file_path)
    except Exception as e:
        logger.exception(f"Failed profiling for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=422, detail="Could not parse dataset")

    return build_dataset_profile(
        df,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        target_column=dataset.target_column,
    )


@router.get("/datasets/{dataset_id}/leakage-report")
def dataset_leakage_report(dataset_id: int, db: Session = Depends(get_db)):
    """Return target leakage risk indicators for a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    try:
        df = _load_dataset_file(dataset.file_path)
    except Exception as e:
        logger.exception(f"Failed leakage report for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=422, detail="Could not parse dataset")

    profile = build_dataset_profile(
        df,
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        target_column=dataset.target_column,
    )
    return {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "target_column": dataset.target_column,
        "leakage_risks": profile.get("leakage_risks", []),
    }


@router.get("/datasets/{dataset_id}/drift-baseline")
def dataset_drift_baseline(dataset_id: int, db: Session = Depends(get_db)):
    """Return baseline distribution snapshot for future drift monitoring."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    try:
        df = _load_dataset_file(dataset.file_path)
    except Exception as e:
        logger.exception(f"Failed drift baseline for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=422, detail="Could not parse dataset")

    return {
        "dataset_id": dataset.id,
        "dataset_name": dataset.name,
        "target_column": dataset.target_column,
        "drift_baseline": build_drift_baseline_snapshot(df, target_column=dataset.target_column),
    }


@router.get("/datasets/{dataset_id}/eda-report")
def dataset_eda_report(dataset_id: int, db: Session = Depends(get_db)):
    """Return structured EDA report for deep dataset understanding."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    try:
        df = _load_dataset_file(dataset.file_path)
    except Exception as e:
        logger.exception(f"Failed EDA report for dataset {dataset_id}: {e}")
        raise HTTPException(status_code=422, detail="Could not parse dataset")

    return {
        "dataset_id": dataset.id,
        **build_eda_report(df, dataset_name=dataset.name, target_column=dataset.target_column),
    }


@router.get("/datasets/{dataset_id}/analytics-report")
def dataset_analytics_report(
    dataset_id: int,
    target_column: Optional[str] = None,
    model_type: Optional[str] = None,
    model_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """Return unified analytics report with >=10 charts and artifact metadata.

    Report is split into:
    1) Exploratory charts
    2) Model evaluation charts

    If model_id is provided, predictions are computed from the saved model artifact and
    evaluation charts are generated; otherwise evaluation is marked pending.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not os.path.exists(dataset.file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    try:
        df = _load_dataset_file(dataset.file_path)
    except Exception as e:
        logger.exception("Failed analytics report for dataset %s: %s", dataset_id, e)
        raise HTTPException(status_code=422, detail="Could not parse dataset")

    resolved_target = target_column or dataset.target_column
    if not resolved_target:
        raise HTTPException(status_code=400, detail="target_column is required when dataset has no default target")
    if resolved_target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{resolved_target}' not found in dataset")

    predictions = None
    resolved_model_type = model_type

    if model_id is not None:
        model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        if not model.file_path or not os.path.exists(model.file_path):
            raise HTTPException(status_code=404, detail="Model artifact not found")

        try:
            bundle = joblib.load(model.file_path)
            predictions = build_predictions_from_model_bundle(
                df,
                target_column=resolved_target,
                model_bundle=bundle,
            )
            resolved_model_type = resolved_model_type or model.task_type
        except Exception as exc:
            logger.warning("Could not generate predictions for analytics report | model_id=%s reason=%s", model_id, exc)
            predictions = None

    report = analytics_service.generate_report(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        df=df,
        target_column=resolved_target,
        model_type=resolved_model_type,
        predictions=predictions,
    )

    report.update(
        {
            "model_id": model_id,
            "model_type": resolved_model_type,
            "predictions_available": bool(predictions is not None),
        }
    )
    return report


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Delete a dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)

    db.delete(dataset)
    db.commit()
    return {"message": "Dataset deleted successfully"}
