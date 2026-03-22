"""Dataset profiling helpers with compact chart-ready outputs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        val = float(v)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except Exception:
        return None


def _dtype_group(dtype: Any) -> str:
    name = str(dtype)
    if name.startswith("int") or name.startswith("float"):
        return "numeric"
    if "datetime" in name:
        return "datetime"
    if name in {"bool", "boolean"}:
        return "boolean"
    return "categorical"


def _histogram(series: pd.Series, bins: int = 20) -> Dict[str, Any]:
    s = series.dropna()
    if len(s) == 0:
        return {"labels": [], "counts": []}

    counts, edges = np.histogram(s.to_numpy(dtype=float), bins=min(max(bins, 5), 30))
    labels = [f"{round(edges[i], 4)} to {round(edges[i + 1], 4)}" for i in range(len(edges) - 1)]
    return {
        "labels": labels,
        "counts": [int(c) for c in counts.tolist()],
    }


def build_dataset_profile(df: pd.DataFrame, dataset_id: int, dataset_name: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    total_rows, total_columns = int(df.shape[0]), int(df.shape[1])
    memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

    missing_total = int(df.isna().sum().sum())
    missing_rate = float(missing_total / max(total_rows * max(total_columns, 1), 1))
    duplicate_rows = int(df.duplicated().sum())

    columns: List[Dict[str, Any]] = []
    missing_by_column: List[Dict[str, Any]] = []
    histograms: List[Dict[str, Any]] = []

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for col in df.columns:
        s = df[col]
        dtype_group = _dtype_group(s.dtype)
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))

        item: Dict[str, Any] = {
            "name": str(col),
            "dtype": str(s.dtype),
            "dtype_group": dtype_group,
            "missing": missing,
            "missing_rate": float(missing / max(total_rows, 1)),
            "unique": unique,
            "unique_rate": float(unique / max(total_rows, 1)),
            "sample_values": [str(v) for v in s.dropna().head(5).tolist()],
        }

        if dtype_group == "numeric":
            sn = s.dropna()
            item["summary"] = {
                "min": _safe_float(sn.min()) if len(sn) > 0 else None,
                "max": _safe_float(sn.max()) if len(sn) > 0 else None,
                "mean": _safe_float(sn.mean()) if len(sn) > 0 else None,
                "std": _safe_float(sn.std()) if len(sn) > 0 else None,
                "median": _safe_float(sn.median()) if len(sn) > 0 else None,
                "q1": _safe_float(sn.quantile(0.25)) if len(sn) > 0 else None,
                "q3": _safe_float(sn.quantile(0.75)) if len(sn) > 0 else None,
            }
            histograms.append({"column": str(col), **_histogram(s)})

        columns.append(item)
        missing_by_column.append({"column": str(col), "missing": missing})

    corr_payload: Optional[Dict[str, Any]] = None
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(numeric_only=True).fillna(0.0)
        corr_payload = {
            "labels": [str(c) for c in corr_df.columns.tolist()],
            "matrix": [[float(v) for v in row] for row in corr_df.to_numpy().tolist()],
        }

    target_distribution = None
    if target_column and target_column in df.columns:
        target = df[target_column]
        if _dtype_group(target.dtype) == "numeric":
            target_distribution = {
                "type": "numeric",
                "column": target_column,
                **_histogram(target),
            }
        else:
            vc = target.astype(str).fillna("<missing>").value_counts().head(20)
            target_distribution = {
                "type": "categorical",
                "column": target_column,
                "labels": vc.index.astype(str).tolist(),
                "counts": [int(v) for v in vc.values.tolist()],
            }

    summary_cards = [
        {"label": "Rows", "value": total_rows},
        {"label": "Columns", "value": total_columns},
        {"label": "Missing cells", "value": missing_total},
        {"label": "Missing rate", "value": round(missing_rate * 100, 2)},
        {"label": "Duplicate rows", "value": duplicate_rows},
        {"label": "Memory (MB)", "value": round(memory_mb, 2)},
    ]

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "memory_usage_mb": round(memory_mb, 4),
        "missing_total": missing_total,
        "missing_rate": round(missing_rate, 8),
        "duplicate_rows": duplicate_rows,
        "columns": columns,
        "summary_cards": summary_cards,
        "target_distribution": target_distribution,
        "correlation_heatmap": corr_payload,
        "missing_by_column": sorted(missing_by_column, key=lambda x: x["missing"], reverse=True),
        "histograms": histograms[:20],
    }
