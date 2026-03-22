"""Utilities to build chart-ready metrics payloads for leaderboard and model analytics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    except Exception:
        return None


def _build_confusion_chart(cm: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(cm, list) or len(cm) == 0:
        return None

    matrix = []
    for row in cm:
        if not isinstance(row, list):
            return None
        matrix.append([int(v) for v in row])

    if len(matrix) == 0:
        return None

    labels = [f"Class {i}" for i in range(len(matrix))]
    normalized = []
    for row in matrix:
        total = sum(row)
        if total <= 0:
            normalized.append([0.0 for _ in row])
        else:
            normalized.append([round(v / total, 6) for v in row])

    return {
        "labels": labels,
        "matrix": matrix,
        "normalized_matrix": normalized,
    }


def _build_roc_chart(roc_data: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(roc_data, dict):
        return None

    fpr = roc_data.get("fpr")
    tpr = roc_data.get("tpr")
    thresholds = roc_data.get("thresholds")

    if not isinstance(fpr, list) or not isinstance(tpr, list) or len(fpr) == 0 or len(fpr) != len(tpr):
        return None

    payload = {
        "fpr": [_safe_float(v) or 0.0 for v in fpr],
        "tpr": [_safe_float(v) or 0.0 for v in tpr],
    }
    if isinstance(thresholds, list) and len(thresholds) == len(fpr):
        payload["thresholds"] = [_safe_float(v) for v in thresholds]
    return payload


def _extract_per_class(metrics: Dict[str, Any]) -> Dict[str, Any]:
    report = metrics.get("classification_report") if isinstance(metrics, dict) else None
    if not isinstance(report, dict):
        return {}

    per_class = {}
    for label, values in report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        if isinstance(values, dict):
            per_class[label] = {
                "precision": _safe_float(values.get("precision")),
                "recall": _safe_float(values.get("recall")),
                "f1-score": _safe_float(values.get("f1-score")),
                "support": int(values.get("support", 0)),
            }
    return per_class


def build_metrics_payload(job_id: str, task_type: str, status: str, models: List[Any], dataset_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    enriched_models: List[Dict[str, Any]] = []

    for model in models:
        metrics = model.metrics if isinstance(model.metrics, dict) else {}

        core = {
            "accuracy": _safe_float(model.accuracy),
            "f1_score": _safe_float(model.f1_score),
            "roc_auc": _safe_float(model.roc_auc),
            "precision": _safe_float(metrics.get("precision")),
            "recall": _safe_float(metrics.get("recall")),
            "balanced_accuracy": _safe_float(metrics.get("balanced_accuracy")),
            "rmse": _safe_float(model.rmse),
            "mae": _safe_float(model.mae),
            "r2_score": _safe_float(model.r2_score),
            "explained_variance": _safe_float(metrics.get("explained_variance")),
            "median_ae": _safe_float(metrics.get("median_ae")),
            "mape": _safe_float(metrics.get("mape")),
        }

        payload = {
            "id": model.id,
            "model_name": model.model_name,
            "model_type": model.model_type,
            "task_type": model.task_type,
            "training_time": _safe_float(model.training_time),
            "is_deployed": bool(model.is_deployed),
            "metrics": {**metrics, **{k: v for k, v in core.items() if v is not None}},
            "cv_scores": model.cv_scores if isinstance(model.cv_scores, list) else None,
            "per_fold": metrics.get("per_fold") if isinstance(metrics.get("per_fold"), list) else None,
            "per_class": _extract_per_class(metrics),
            "feature_importance": model.feature_importance if isinstance(model.feature_importance, dict) else None,
            "confusion_matrix_chart": _build_confusion_chart(model.confusion_matrix),
            "roc_curve_chart": _build_roc_chart(model.roc_curve_data),
            # backward-compatible fields
            "accuracy": core["accuracy"],
            "f1_score": core["f1_score"],
            "roc_auc": core["roc_auc"],
            "precision": core["precision"],
            "recall": core["recall"],
            "balanced_accuracy": core["balanced_accuracy"],
            "rmse": core["rmse"],
            "mae": core["mae"],
            "r2_score": core["r2_score"],
            "mape": core["mape"],
            "explained_variance": core["explained_variance"],
            "median_ae": core["median_ae"],
            "confusion_matrix": model.confusion_matrix,
            "roc_curve_data": model.roc_curve_data,
            "cv_scores": model.cv_scores,
        }
        enriched_models.append(payload)

    primary_metric = "accuracy" if task_type == "classification" else "r2_score"

    def _metric_for_sort(m: Dict[str, Any]) -> float:
        v = _safe_float((m.get("metrics") or {}).get(primary_metric))
        return v if v is not None else -1e9

    enriched_models.sort(key=_metric_for_sort, reverse=True)

    leaderboard = [
        {
            "rank": idx + 1,
            "model_id": m["id"],
            "model_name": m["model_name"],
            "primary_metric": (m.get("metrics") or {}).get(primary_metric),
            "metric_name": primary_metric,
            "training_time": m.get("training_time"),
            "is_deployed": m.get("is_deployed"),
        }
        for idx, m in enumerate(enriched_models)
    ]

    metric_catalog = sorted(
        {
            k
            for m in enriched_models
            for k, v in (m.get("metrics") or {}).items()
            if isinstance(v, (int, float))
        }
    )

    labels = [m["model_name"] for m in enriched_models]
    comparison_series: Dict[str, List[Optional[float]]] = {}
    for metric_name in metric_catalog:
        comparison_series[metric_name] = [
            _safe_float((m.get("metrics") or {}).get(metric_name))
            for m in enriched_models
        ]

    roc_curves = [
        {
            "model_id": m["id"],
            "model_name": m["model_name"],
            "curve": m.get("roc_curve_chart"),
        }
        for m in enriched_models
        if m.get("roc_curve_chart") is not None
    ]

    confusion_matrices = [
        {
            "model_id": m["id"],
            "model_name": m["model_name"],
            "matrix": m.get("confusion_matrix_chart"),
        }
        for m in enriched_models
        if m.get("confusion_matrix_chart") is not None
    ]

    best_model_id = enriched_models[0]["id"] if len(enriched_models) > 0 else None

    return {
        "job_id": job_id,
        "task_type": task_type,
        "status": status,
        "metric_catalog": metric_catalog,
        "models": enriched_models,
        "best_model_id": best_model_id,
        "best_model": enriched_models[0] if len(enriched_models) > 0 else None,
        "leaderboard": leaderboard,
        "chart_data": {
            "metric_comparison": {
                "labels": labels,
                "series": comparison_series,
            },
            "roc_curves": roc_curves,
            "confusion_matrices": confusion_matrices,
        },
        "dataset_profile": dataset_profile,
    }
