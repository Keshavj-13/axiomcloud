"""Dataset profiling helpers with compact chart-ready outputs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

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


def _infer_semantic_type(series: pd.Series, dtype_group: str, total_rows: int) -> Dict[str, Any]:
    unique = int(series.nunique(dropna=True))
    unique_rate = float(unique / max(total_rows, 1))

    if dtype_group == "numeric":
        return {
            "semantic_type": "continuous" if unique_rate > 0.1 else "ordinal_numeric",
            "datetime_parse_ratio": 0.0,
        }

    if dtype_group == "datetime":
        return {
            "semantic_type": "datetime",
            "datetime_parse_ratio": 1.0,
        }

    if dtype_group == "boolean":
        return {
            "semantic_type": "boolean",
            "datetime_parse_ratio": 0.0,
        }

    # Categorical/text heuristics
    s = series.dropna().astype(str)
    dt_ratio = 0.0
    if len(s) > 0:
        parsed = pd.to_datetime(s.sample(min(len(s), 300), random_state=42), errors="coerce", utc=False)
        dt_ratio = float(parsed.notna().mean())

    semantic = "nominal"
    if dt_ratio >= 0.8:
        semantic = "datetime_like"
    elif unique <= 20:
        semantic = "ordinal_or_nominal_low_cardinality"

    return {
        "semantic_type": semantic,
        "datetime_parse_ratio": round(dt_ratio, 4),
    }


def _detect_target_leakage_risk(df: pd.DataFrame, target_column: Optional[str]) -> List[Dict[str, Any]]:
    if not target_column or target_column not in df.columns:
        return []

    target = df[target_column]
    risks: List[Dict[str, Any]] = []

    # Name-based leakage hints
    target_tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", str(target_column).lower()) if t]

    for col in df.columns:
        if col == target_column:
            continue

        col_lower = str(col).lower()
        token_overlap = any(tok in col_lower for tok in target_tokens if len(tok) >= 3)

        risk_score = 0.0
        reasons: List[str] = []

        if token_overlap:
            risk_score += 0.2
            reasons.append("feature name overlaps with target name")

        # Exact/near-exact equality check (categorical leakage)
        try:
            pair = pd.DataFrame({"a": df[col], "b": target}).dropna()
            if len(pair) > 0:
                eq_rate = float((pair["a"].astype(str) == pair["b"].astype(str)).mean())
                if eq_rate >= 0.98:
                    risk_score += 0.9
                    reasons.append(f"feature matches target values in {eq_rate:.2%} rows")
        except Exception:
            pass

        # High correlation check for numeric target/feature
        try:
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target):
                corr = float(pd.concat([df[col], target], axis=1).corr(numeric_only=True).iloc[0, 1])
                if np.isfinite(corr) and abs(corr) >= 0.98:
                    risk_score += 0.8
                    reasons.append(f"very high absolute correlation with target ({corr:.4f})")
        except Exception:
            pass

        if risk_score > 0:
            risks.append(
                {
                    "feature": str(col),
                    "risk_score": round(min(risk_score, 1.0), 4),
                    "reasons": reasons,
                }
            )

    return sorted(risks, key=lambda x: x["risk_score"], reverse=True)[:20]


def build_drift_baseline_snapshot(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Build compact baseline distribution snapshot for future drift comparisons."""
    baseline: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric": {},
        "categorical": {},
        "target_column": target_column,
    }

    for col in df.columns:
        if target_column and col == target_column:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            sn = s.dropna()
            baseline["numeric"][str(col)] = {
                "mean": _safe_float(sn.mean()) if len(sn) > 0 else None,
                "std": _safe_float(sn.std()) if len(sn) > 0 else None,
                "q05": _safe_float(sn.quantile(0.05)) if len(sn) > 0 else None,
                "q50": _safe_float(sn.quantile(0.50)) if len(sn) > 0 else None,
                "q95": _safe_float(sn.quantile(0.95)) if len(sn) > 0 else None,
                "missing_rate": float(s.isna().mean()),
            }
        else:
            vc = s.astype(str).fillna("<missing>").value_counts(normalize=True).head(10)
            baseline["categorical"][str(col)] = {
                "top_values": [{"value": str(k), "rate": float(v)} for k, v in vc.items()],
                "missing_rate": float(s.isna().mean()),
            }

    return baseline


def build_eda_report(df: pd.DataFrame, dataset_name: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """Build a structured EDA report payload for UI consumption."""
    total_rows, total_columns = int(df.shape[0]), int(df.shape[1])
    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    numeric_summary: Dict[str, Any] = {}
    for col in numeric_cols[:40]:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        numeric_summary[str(col)] = {
            "count": int(len(s)),
            "mean": _safe_float(s.mean()),
            "std": _safe_float(s.std()),
            "min": _safe_float(s.min()),
            "q1": _safe_float(s.quantile(0.25)),
            "median": _safe_float(s.quantile(0.50)),
            "q3": _safe_float(s.quantile(0.75)),
            "max": _safe_float(s.max()),
            "histogram": _histogram(s),
        }

    categorical_summary: Dict[str, Any] = {}
    for col in categorical_cols[:40]:
        s = df[col].astype(str).fillna("<missing>")
        vc = s.value_counts().head(20)
        categorical_summary[str(col)] = {
            "unique": int(df[col].nunique(dropna=True)),
            "top_values": [{"value": str(k), "count": int(v)} for k, v in vc.items()],
        }

    correlations: List[Dict[str, Any]] = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).fillna(0.0)
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = _safe_float(corr.iloc[i, j])
                if v is None:
                    continue
                if abs(v) >= 0.75:
                    correlations.append(
                        {
                            "feature_a": str(cols[i]),
                            "feature_b": str(cols[j]),
                            "correlation": float(v),
                        }
                    )
    correlations = sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)[:30]

    leakage_risks = _detect_target_leakage_risk(df, target_column=target_column)

    recommendations: List[str] = []
    if missing_total > 0:
        recommendations.append("Impute or remove missing values before final training runs.")
    if duplicate_rows > 0:
        recommendations.append("Drop duplicate rows to avoid biased validation metrics.")
    if len(correlations) > 0:
        recommendations.append("Review highly correlated features for redundancy and leakage risk.")
    if len(leakage_risks) > 0:
        recommendations.append("Investigate high-risk leakage features before modeling.")
    if not recommendations:
        recommendations.append("Dataset looks stable for modeling; proceed with tuned training.")

    return {
        "dataset_name": dataset_name,
        "overview": {
            "rows": total_rows,
            "columns": total_columns,
            "missing_total": missing_total,
            "duplicate_rows": duplicate_rows,
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "target_column": target_column,
        },
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "high_correlations": correlations,
        "leakage_risks": leakage_risks,
        "typing_intelligence": {
            "semantic_types": [
                {
                    "column": str(c),
                    **_infer_semantic_type(df[c], _dtype_group(df[c].dtype), total_rows),
                }
                for c in df.columns[:120]
            ]
        },
        "drift_baseline": build_drift_baseline_snapshot(df, target_column=target_column),
        "recommendations": recommendations,
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
        item.update(_infer_semantic_type(s, dtype_group=dtype_group, total_rows=total_rows))

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

    leakage_risks = _detect_target_leakage_risk(df, target_column=target_column)

    typing_intelligence = {
        "numeric": int(sum(1 for c in columns if c.get("dtype_group") == "numeric")),
        "categorical": int(sum(1 for c in columns if c.get("dtype_group") == "categorical")),
        "datetime": int(sum(1 for c in columns if c.get("dtype_group") == "datetime" or c.get("semantic_type") == "datetime_like")),
        "boolean": int(sum(1 for c in columns if c.get("dtype_group") == "boolean")),
        "high_cardinality_candidates": [
            c["name"] for c in columns if c.get("dtype_group") == "categorical" and (c.get("unique_rate") or 0) > 0.3
        ][:20],
    }

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
        "typing_intelligence": typing_intelligence,
        "leakage_risks": leakage_risks,
        "drift_baseline": build_drift_baseline_snapshot(df, target_column=target_column),
        "missing_by_column": sorted(missing_by_column, key=lambda x: x["missing"], reverse=True),
        "histograms": histograms[:20],
    }
