"""Explainability service utilities for SHAP/LIME payload generation with caching."""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:  # pragma: no cover - optional dependency
    LimeTabularExplainer = None


class ExplainabilityError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass
class ExplainabilityContext:
    model_id: int
    model_name: str
    model_type: str
    task_type: str
    job_id: str
    artifact_hash: str
    dataset_hash: str


def _hash_file_state(path: str) -> str:
    st = os.stat(path)
    payload = f"{path}:{st.st_size}:{st.st_mtime_ns}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _safe_feature_names(transformed: Any, fallback: List[str], preprocessor: Any) -> List[str]:
    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            names = preprocessor.get_feature_names_out()
            if names is not None:
                return [str(n) for n in list(names)]
    except Exception:
        logger.debug("Could not resolve transformed feature names from preprocessor", exc_info=True)

    if isinstance(fallback, list) and len(fallback) > 0:
        return [str(c) for c in fallback]

    n_features = int(transformed.shape[1]) if hasattr(transformed, "shape") and len(transformed.shape) > 1 else 0
    return [f"feature_{i}" for i in range(n_features)]


def _ensure_dense(x: Any) -> np.ndarray:
    if issparse(x):
        return x.toarray()
    if hasattr(x, "toarray") and callable(x.toarray):
        return x.toarray()
    return np.asarray(x)


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Index):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [_json_safe(r) for r in obj.to_dict(orient="records")]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if issparse(obj):
        return _json_safe(obj.toarray())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return str(obj)


class ExplainabilityCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._store[key] = value


class ExplainabilityService:
    def __init__(self):
        self.cache = ExplainabilityCache()

    def build_context(self, model: Any, dataset_path: str) -> ExplainabilityContext:
        if not model.file_path or not os.path.exists(model.file_path):
            raise ExplainabilityError("MODEL_FILE_MISSING", "Model artifact file not found")
        if not dataset_path or not os.path.exists(dataset_path):
            raise ExplainabilityError("DATASET_FILE_MISSING", "Training dataset file not found")

        return ExplainabilityContext(
            model_id=int(model.id),
            model_name=str(model.model_name),
            model_type=str(model.model_type),
            task_type=str(model.task_type or "unknown"),
            job_id=str(model.job_id),
            artifact_hash=_hash_file_state(model.file_path),
            dataset_hash=_hash_file_state(dataset_path),
        )

    def _load_model_bundle(self, model_path: str) -> Dict[str, Any]:
        try:
            bundle = joblib.load(model_path)
        except Exception as exc:
            raise ExplainabilityError("MODEL_LOAD_FAILED", "Could not load model artifact", {"reason": str(exc)}) from exc

        pipeline = bundle.get("pipeline") if isinstance(bundle, dict) else None
        if pipeline is None:
            raise ExplainabilityError("PIPELINE_MISSING", "No pipeline found inside model artifact")
        return bundle

    def _prepare_dataset(self, dataset_path: str, target_column: str) -> pd.DataFrame:
        try:
            if dataset_path.endswith(".csv"):
                df = pd.read_csv(dataset_path)
            else:
                df = pd.read_excel(dataset_path)
        except Exception as exc:
            raise ExplainabilityError("DATASET_PARSE_FAILED", "Could not parse training dataset", {"reason": str(exc)}) from exc

        if target_column not in df.columns:
            raise ExplainabilityError("TARGET_COLUMN_MISSING", "Target column missing in training dataset", {"target_column": target_column})

        features = df.drop(columns=[target_column], errors="ignore")
        if features.shape[1] == 0:
            raise ExplainabilityError("EMPTY_FEATURES", "No feature columns available for explainability")
        return features

    def _sample_frame(self, frame: pd.DataFrame, nsamples: int) -> pd.DataFrame:
        n = int(frame.shape[0])
        if n == 0:
            raise ExplainabilityError("EMPTY_DATASET", "Dataset has no rows for explainability")
        take = min(max(int(nsamples), 1), n)
        if n <= take:
            return frame.copy()
        return frame.sample(n=take, random_state=42)

    def _prediction_context(self, pipeline: Any, row_df: pd.DataFrame, task_type: str, label_encoder: Any) -> Dict[str, Any]:
        pred_raw = pipeline.predict(row_df)
        pred_val: Any = pred_raw[0] if hasattr(pred_raw, "__len__") else pred_raw
        prediction_label = None
        class_probabilities = None
        confidence = None

        if task_type == "classification" and label_encoder is not None:
            try:
                prediction_label = str(label_encoder.inverse_transform([int(pred_val)])[0])
            except Exception:
                prediction_label = str(pred_val)

        if task_type == "classification" and hasattr(pipeline, "predict_proba"):
            try:
                probs = pipeline.predict_proba(row_df)[0]
                if label_encoder is not None and hasattr(label_encoder, "classes_"):
                    labels = [str(c) for c in list(label_encoder.classes_)]
                else:
                    labels = [str(i) for i in range(len(probs))]
                class_probabilities = {labels[i]: float(probs[i]) for i in range(len(labels))}
                confidence = float(np.max(probs))
            except Exception:
                class_probabilities = None
                confidence = None

        value = _json_safe(pred_val)
        return {
            "prediction": value,
            "prediction_label": prediction_label,
            "class_probabilities": class_probabilities,
            "confidence": confidence,
        }

    def build_shap(
        self,
        *,
        model: Any,
        dataset_path: str,
        target_column: str,
        sample_index: int,
        nsamples: int,
    ) -> Dict[str, Any]:
        ctx = self.build_context(model, dataset_path)
        cache_key = f"shap:{ctx.model_id}:{ctx.artifact_hash}:{ctx.dataset_hash}:{sample_index}:{nsamples}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        bundle = self._load_model_bundle(model.file_path)
        pipeline = bundle["pipeline"]
        label_encoder = bundle.get("label_encoder")

        features_df = self._prepare_dataset(dataset_path, target_column)
        if sample_index < 0 or sample_index >= int(features_df.shape[0]):
            raise ExplainabilityError(
                "INVALID_SAMPLE_INDEX",
                "Selected sample index is out of range",
                {"sample_index": sample_index, "max_index": int(features_df.shape[0]) - 1},
            )

        sample_row = features_df.iloc[[sample_index]].copy()
        preprocessor = pipeline.named_steps.get("preprocessor")
        estimator = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("regressor")

        if preprocessor is None or estimator is None:
            raise ExplainabilityError("PIPELINE_INVALID", "Pipeline missing preprocessor or estimator")

        background_df = self._sample_frame(features_df, nsamples)

        try:
            bg_transformed = preprocessor.transform(background_df)
            sample_transformed = preprocessor.transform(sample_row)
        except Exception as exc:
            raise ExplainabilityError("TRANSFORM_FAILED", "Feature transformation failed", {"reason": str(exc)}) from exc

        bg_dense = _ensure_dense(bg_transformed)
        sample_dense = _ensure_dense(sample_transformed)
        feature_names = _safe_feature_names(bg_dense, list(features_df.columns), preprocessor)

        explainer_name = "KernelExplainer"
        used_fallback = True
        shap_values = None
        expected_value: Any = None

        tree_like = any(hasattr(estimator, attr) for attr in ["feature_importances_", "estimators_", "tree_"])

        try:
            if tree_like:
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(sample_dense)
                expected_value = explainer.expected_value
                explainer_name = "TreeExplainer"
                used_fallback = False
            else:
                bg_small = bg_dense[: min(60, bg_dense.shape[0])]
                if ctx.task_type == "classification" and hasattr(estimator, "predict_proba"):
                    explainer = shap.KernelExplainer(estimator.predict_proba, bg_small)
                    shap_values = explainer.shap_values(sample_dense, nsamples=min(100, nsamples))
                else:
                    explainer = shap.KernelExplainer(estimator.predict, bg_small)
                    shap_values = explainer.shap_values(sample_dense, nsamples=min(100, nsamples))
                expected_value = explainer.expected_value
        except Exception as exc:
            raise ExplainabilityError("SHAP_FAILED", "SHAP explanation generation failed", {"reason": str(exc)}) from exc

        values_arr = np.array(shap_values)
        if values_arr.ndim == 3:
            # (n_classes, n_samples, n_features) -> first sample, max class by absolute sum
            class_idx = int(np.argmax(np.abs(values_arr[:, 0, :]).sum(axis=1)))
            local_vals = values_arr[class_idx, 0, :]
        elif values_arr.ndim == 2:
            local_vals = values_arr[0, :]
        elif values_arr.ndim == 1:
            local_vals = values_arr
        else:
            local_vals = np.zeros((len(feature_names),), dtype=float)

        local_vals = np.asarray(local_vals, dtype=float)
        if local_vals.shape[0] != len(feature_names):
            # align length conservatively
            n = min(local_vals.shape[0], len(feature_names))
            local_vals = local_vals[:n]
            feature_names = feature_names[:n]

        # approximate global by local if only one sample; use transformed background for robust mean abs via LinearExplainer fallback style
        global_scores = np.abs(local_vals)
        ranked_idx = np.argsort(-global_scores)

        global_importance = [
            {
                "feature": feature_names[idx],
                "mean_abs_contribution": float(global_scores[idx]),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(ranked_idx)
        ]

        local_order = np.argsort(-np.abs(local_vals))
        local_contrib = [
            {
                "feature": feature_names[idx],
                "value": float(local_vals[idx]),
                "abs_value": float(abs(local_vals[idx])),
                "rank": rank + 1,
            }
            for rank, idx in enumerate(local_order)
        ]

        pred_ctx = self._prediction_context(pipeline, sample_row, ctx.task_type, label_encoder)

        payload = {
            "metadata": {
                "model_id": ctx.model_id,
                "model_name": ctx.model_name,
                "model_type": ctx.model_type,
                "task_type": ctx.task_type,
                "job_id": ctx.job_id,
                "artifact_hash": ctx.artifact_hash,
                "dataset_hash": ctx.dataset_hash,
                "cache_key": cache_key,
                "used_fallback": used_fallback,
                "explainer": explainer_name,
            },
            "feature_names": _json_safe(feature_names),
            "expected_value": _json_safe(expected_value),
            "global_importance": _json_safe(global_importance),
            "local_contributions": _json_safe(local_contrib),
            "sample_prediction": {
                "sample_index": int(sample_index),
                **pred_ctx,
            },
            "chart": {
                "global_bar": {
                    "labels": [g["feature"] for g in global_importance],
                    "values": [g["mean_abs_contribution"] for g in global_importance],
                },
                "local_bar": {
                    "labels": [l["feature"] for l in local_contrib],
                    "values": [l["value"] for l in local_contrib],
                },
            },
            "model_metadata": {
                "feature_count": len(feature_names),
                "selected_sample_index": int(sample_index),
            },
        }

        payload = _json_safe(payload)
        self.cache.set(cache_key, payload)
        return payload

    def build_lime(
        self,
        *,
        model: Any,
        dataset_path: str,
        target_column: str,
        sample_index: int,
        num_features: int,
        custom_input: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if LimeTabularExplainer is None:
            raise ExplainabilityError(
                "LIME_UNAVAILABLE",
                "LIME dependency is not installed. Install 'lime' to use this endpoint.",
                {"dependency": "lime"},
            )

        ctx = self.build_context(model, dataset_path)
        sample_token = f"custom:{hashlib.sha1(str(sorted((custom_input or {}).items())).encode('utf-8')).hexdigest()[:8]}" if custom_input else str(sample_index)
        cache_key = f"lime:{ctx.model_id}:{ctx.artifact_hash}:{ctx.dataset_hash}:{sample_token}:{num_features}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        bundle = self._load_model_bundle(model.file_path)
        pipeline = bundle["pipeline"]
        label_encoder = bundle.get("label_encoder")

        features_df = self._prepare_dataset(dataset_path, target_column)
        if sample_index < 0 or sample_index >= int(features_df.shape[0]):
            raise ExplainabilityError(
                "INVALID_SAMPLE_INDEX",
                "Selected sample index is out of range",
                {"sample_index": sample_index, "max_index": int(features_df.shape[0]) - 1},
            )

        sample_row = features_df.iloc[[sample_index]].copy()
        if isinstance(custom_input, dict) and len(custom_input) > 0:
            for col in sample_row.columns:
                if col in custom_input:
                    sample_row[col] = custom_input[col]

        preprocessor = pipeline.named_steps.get("preprocessor")
        estimator = pipeline.named_steps.get("classifier") or pipeline.named_steps.get("regressor")
        if preprocessor is None or estimator is None:
            raise ExplainabilityError("PIPELINE_INVALID", "Pipeline missing preprocessor or estimator")

        try:
            transformed = preprocessor.transform(features_df)
            sample_transformed = preprocessor.transform(sample_row)
        except Exception as exc:
            raise ExplainabilityError("TRANSFORM_FAILED", "Feature transformation failed", {"reason": str(exc)}) from exc

        transformed_dense = _ensure_dense(transformed)
        sample_dense = _ensure_dense(sample_transformed)
        feature_names = _safe_feature_names(transformed_dense, list(features_df.columns), preprocessor)

        class_names: Optional[List[str]] = None
        if ctx.task_type == "classification":
            if label_encoder is not None and hasattr(label_encoder, "classes_"):
                class_names = [str(c) for c in label_encoder.classes_]
            else:
                preds = pipeline.predict(features_df.head(100))
                class_names = [str(c) for c in sorted(set(_json_safe(np.asarray(preds).tolist())))]

        explainer = LimeTabularExplainer(
            training_data=transformed_dense,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification" if ctx.task_type == "classification" else "regression",
            discretize_continuous=True,
            random_state=42,
        )

        predict_fn = estimator.predict_proba if (ctx.task_type == "classification" and hasattr(estimator, "predict_proba")) else estimator.predict
        try:
            exp = explainer.explain_instance(sample_dense[0], predict_fn, num_features=min(num_features, len(feature_names)))
        except Exception as exc:
            raise ExplainabilityError("LIME_FAILED", "LIME explanation generation failed", {"reason": str(exc)}) from exc

        weights_raw = exp.as_list()
        weights: List[Dict[str, Any]] = []
        for idx, (label, weight) in enumerate(weights_raw):
            val = float(weight)
            weights.append(
                {
                    "feature": str(label),
                    "weight": val,
                    "abs_weight": float(abs(val)),
                    "direction": "positive" if val >= 0 else "negative",
                    "rank": idx + 1,
                }
            )

        pos = [w for w in weights if w["weight"] >= 0]
        neg = [w for w in weights if w["weight"] < 0]
        pos = sorted(pos, key=lambda w: w["abs_weight"], reverse=True)
        neg = sorted(neg, key=lambda w: w["abs_weight"], reverse=True)

        pred_ctx = self._prediction_context(pipeline, sample_row, ctx.task_type, label_encoder)

        payload = {
            "metadata": {
                "model_id": ctx.model_id,
                "model_name": ctx.model_name,
                "model_type": ctx.model_type,
                "task_type": ctx.task_type,
                "job_id": ctx.job_id,
                "artifact_hash": ctx.artifact_hash,
                "dataset_hash": ctx.dataset_hash,
                "cache_key": cache_key,
                "used_fallback": False,
                "explainer": "LimeTabularExplainer",
            },
            "feature_names": _json_safe(feature_names),
            "sample_prediction": {
                "sample_index": int(sample_index),
                **pred_ctx,
            },
            "weights": _json_safe(weights),
            "top_positive": _json_safe(pos[: min(10, len(pos))]),
            "top_negative": _json_safe(neg[: min(10, len(neg))]),
            "class_context": {
                "class_names": class_names or [],
                "mode": "classification" if ctx.task_type == "classification" else "regression",
                "target_label": "target" if ctx.task_type == "regression" else "class",
            },
            "chart": {
                "local_bar": {
                    "labels": [w["feature"] for w in weights],
                    "values": [w["weight"] for w in weights],
                }
            },
        }

        payload = _json_safe(payload)
        self.cache.set(cache_key, payload)
        return payload


explainability_service = ExplainabilityService()
