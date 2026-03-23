"""Analytics report service for EDA + model evaluation diagnostics.

Generates chart metadata (with persisted artifact files) for:
1) Exploratory data analysis
2) Model evaluation

Rules enforced:
- At least 10 charts per report
- Task auto-detection from target dtype/profile
- Fallback chart generated when a required chart fails
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from app.core.config import settings


MIN_CHARTS = 10


@dataclass
class ChartContext:
    dataset_id: int
    report_id: str
    reports_root: str


class AnalyticsReportService:
    def __init__(self) -> None:
        self.reports_root = os.path.abspath(os.path.join(settings.BASE_STORAGE_PATH, "reports"))
        os.makedirs(self.reports_root, exist_ok=True)

    def detect_task_type(self, target: pd.Series) -> str:
        """Rule-based task detection.

        - categorical target => classification
        - continuous numeric target => regression
        """
        if not pd.api.types.is_numeric_dtype(target):
            return "classification"

        clean = target.dropna()
        unique = int(clean.nunique())
        unique_rate = float(unique / max(len(clean), 1))

        # Numeric but low-cardinality target usually means encoded classes.
        if unique <= 20 and unique_rate <= 0.2:
            return "classification"
        return "regression"

    def generate_report(
        self,
        *,
        dataset_id: int,
        dataset_name: str,
        df: pd.DataFrame,
        target_column: str,
        model_type: Optional[str] = None,
        predictions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        report_id = f"{dataset_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        ctx = ChartContext(
            dataset_id=dataset_id,
            report_id=report_id,
            reports_root=self.reports_root,
        )

        target = df[target_column]
        detected_task = self.detect_task_type(target)
        task_type = detected_task if model_type in {None, "", "auto"} else model_type

        exploratory: List[Dict[str, Any]] = []
        evaluation: List[Dict[str, Any]] = []
        evaluation_metrics: Dict[str, Any] = {}
        evaluation_status = "pending"

        if task_type == "classification":
            exploratory = self._build_classification_eda(ctx, df, target_column)
            if predictions and predictions.get("y_true") is not None and predictions.get("y_pred") is not None:
                evaluation, evaluation_metrics = self._build_classification_eval(
                    ctx,
                    np.asarray(predictions["y_true"]),
                    np.asarray(predictions["y_pred"]),
                )
                evaluation_status = "ready"
            else:
                evaluation = [
                    self._fallback_chart(
                        ctx,
                        chart_id="classification_evaluation_pending",
                        section="model_evaluation",
                        title="Classification evaluation pending",
                        purpose="Model-level class evaluation charts require predictions.",
                        insight="Predictions are not available yet, so evaluation is pending.",
                        fallback_payload={
                            "rows": [
                                {"label": "status", "value": "pending"},
                                {"label": "required", "value": "y_true and y_pred"},
                            ]
                        },
                    )
                ]
        else:
            exploratory = self._build_regression_eda(ctx, df, target_column)
            if predictions and predictions.get("y_true") is not None and predictions.get("y_pred") is not None:
                y_true = np.asarray(predictions["y_true"], dtype=float)
                y_pred = np.asarray(predictions["y_pred"], dtype=float)
                evaluation, evaluation_metrics = self._build_regression_eval(ctx, y_true, y_pred)
                evaluation_status = "ready"
            else:
                evaluation = [
                    self._fallback_chart(
                        ctx,
                        chart_id="regression_evaluation_pending",
                        section="model_evaluation",
                        title="Regression evaluation pending",
                        purpose="Residual and parity plots require predictions.",
                        insight="Predictions are not available yet, so evaluation charts are pending.",
                        fallback_payload={
                            "rows": [
                                {"label": "status", "value": "pending"},
                                {"label": "required", "value": "y_true and y_pred"},
                            ]
                        },
                    )
                ]

        charts = exploratory + evaluation
        if len(charts) < MIN_CHARTS:
            charts.extend(self._padding_charts(ctx, MIN_CHARTS - len(charts)))

        # Keep sections separated for frontend.
        exploratory_final = [c for c in charts if c["section"] == "exploratory"]
        evaluation_final = [c for c in charts if c["section"] == "model_evaluation"]

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "target_column": target_column,
            "task_type": task_type,
            "detected_task_type": detected_task,
            "minimum_required_charts": MIN_CHARTS,
            "chart_count": len(charts),
            "report_id": report_id,
            "created_at": datetime.utcnow().isoformat(),
            "exploratory_charts": exploratory_final,
            "evaluation_charts": evaluation_final,
            "evaluation_status": evaluation_status,
            "evaluation_metrics": evaluation_metrics,
        }

    # ---- Classification builders -------------------------------------------------

    def _build_classification_eda(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> List[Dict[str, Any]]:
        charts: List[Dict[str, Any]] = []
        target = df[target_column]

        charts.append(self._chart_data_type_composition(ctx, df))
        charts.append(self._chart_unique_value_density(ctx, df, target_column))
        charts.append(self._chart_numeric_skewness_profile(ctx, df, target_column))
        charts.append(self._chart_numeric_variance_profile(ctx, df, target_column))
        charts.append(self._chart_categorical_concentration(ctx, df, target_column))
        charts.append(self._chart_target_class_distribution(ctx, target))
        charts.append(self._chart_missing_values(ctx, df))
        charts.append(self._chart_numeric_distributions(ctx, df, target_column))
        charts.append(self._chart_boxplot_by_class(ctx, df, target_column))
        charts.append(self._chart_correlation_heatmap(ctx, df, target_column))
        charts.append(self._chart_pairplot_top_numeric(ctx, df, target_column))
        charts.append(self._chart_categorical_vs_class_rate(ctx, df, target_column))
        charts.append(self._chart_feature_ranking(ctx, df, target_column, task_type="classification"))
        charts.append(self._chart_outlier_analysis(ctx, df, target_column))

        return charts

    def _build_classification_eval(
        self,
        ctx: ChartContext,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        labels = sorted([str(v) for v in np.unique(np.concatenate([y_true.astype(str), y_pred.astype(str)]))])
        cm = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=labels)

        accuracy = float(accuracy_score(y_true.astype(str), y_pred.astype(str)))
        precision = float(precision_score(y_true.astype(str), y_pred.astype(str), average="weighted", zero_division=0))
        recall = float(recall_score(y_true.astype(str), y_pred.astype(str), average="weighted", zero_division=0))
        f1 = float(f1_score(y_true.astype(str), y_pred.astype(str), average="weighted", zero_division=0))

        per_class_rows = []
        for idx, label in enumerate(labels):
            tp = float(cm[idx, idx])
            support = float(cm[idx, :].sum())
            predicted = float(cm[:, idx].sum())
            cls_recall = tp / support if support > 0 else 0.0
            cls_precision = tp / predicted if predicted > 0 else 0.0
            per_class_rows.append(
                {
                    "class": label,
                    "precision": round(cls_precision, 6),
                    "recall": round(cls_recall, 6),
                    "support": int(support),
                }
            )

        charts = [
            self._save_chart(
                ctx,
                chart_id="classification_confusion_matrix",
                section="model_evaluation",
                title="Confusion matrix",
                purpose="Shows class-level prediction outcomes.",
                insight="Use off-diagonal cells to identify confused classes.",
                chart_type="heatmap",
                payload={
                    "xLabels": labels,
                    "yLabels": labels,
                    "matrix": [[int(v) for v in row] for row in cm.tolist()],
                },
            ),
            self._save_chart(
                ctx,
                chart_id="classification_per_class_metrics",
                section="model_evaluation",
                title="Per-class precision and recall",
                purpose="Validates class-level performance consistency.",
                insight="Classes with low recall are frequently missed.",
                chart_type="bar",
                payload={
                    "xKey": "class",
                    "rows": per_class_rows,
                    "series": [
                        {"key": "precision", "name": "Precision", "color": "#d0bcff"},
                        {"key": "recall", "name": "Recall", "color": "#6e3bd7"},
                    ],
                },
            ),
        ]

        metrics = {
            "accuracy": round(accuracy, 6),
            "precision_weighted": round(precision, 6),
            "recall_weighted": round(recall, 6),
            "f1_weighted": round(f1, 6),
        }
        return charts, metrics

    # ---- Regression builders -----------------------------------------------------

    def _build_regression_eda(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> List[Dict[str, Any]]:
        charts: List[Dict[str, Any]] = []
        target = pd.to_numeric(df[target_column], errors="coerce")

        charts.append(self._chart_data_type_composition(ctx, df))
        charts.append(self._chart_unique_value_density(ctx, df, target_column))
        charts.append(self._chart_numeric_skewness_profile(ctx, df, target_column))
        charts.append(self._chart_numeric_variance_profile(ctx, df, target_column))
        charts.append(self._chart_categorical_concentration(ctx, df, target_column))
        charts.append(self._chart_target_distribution(ctx, target))
        charts.append(self._chart_missing_values(ctx, df))
        charts.append(self._chart_numeric_distributions(ctx, df, target_column))
        charts.append(self._chart_feature_vs_target_scatter(ctx, df, target_column))
        charts.append(self._chart_correlation_heatmap(ctx, df, target_column))
        charts.append(self._chart_pairplot_top_numeric(ctx, df, target_column))
        charts.append(self._chart_binned_target_by_category(ctx, df, target_column))
        charts.append(self._chart_feature_ranking(ctx, df, target_column, task_type="regression"))
        charts.append(self._chart_outlier_analysis(ctx, df, target_column))

        return charts

    def _build_regression_eval(
        self,
        ctx: ChartContext,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        residuals = y_true - y_pred

        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))

        abs_res = np.abs(residuals)
        bias = float(np.mean(residuals))
        large_threshold = float(np.percentile(abs_res, 95)) if len(abs_res) > 0 else 0.0
        large_residual_count = int((abs_res >= large_threshold).sum()) if len(abs_res) > 0 else 0

        charts = [
            self._chart_residual_distribution(ctx, residuals),
            self._chart_residuals_vs_predicted(ctx, y_pred, residuals),
            self._chart_actual_vs_predicted(ctx, y_true, y_pred),
            self._save_chart(
                ctx,
                chart_id="regression_error_metrics",
                section="model_evaluation",
                title="Regression error metrics",
                purpose="Summarizes absolute, squared, and explained error.",
                insight=f"Bias={bias:.4f}; large residual points={large_residual_count} (p95 threshold={large_threshold:.4f}).",
                chart_type="table",
                payload={
                    "rows": [
                        {"label": "MAE", "value": round(mae, 6)},
                        {"label": "MSE", "value": round(mse, 6)},
                        {"label": "RMSE", "value": round(rmse, 6)},
                        {"label": "R²", "value": round(r2, 6)},
                        {"label": "Mean residual bias", "value": round(bias, 6)},
                        {"label": "Large residual count", "value": large_residual_count},
                    ]
                },
            ),
        ]

        metrics = {
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "r2": round(r2, 6),
            "mean_residual_bias": round(bias, 6),
            "large_residual_count": large_residual_count,
        }
        return charts, metrics

    # ---- Shared chart builders ---------------------------------------------------

    def _chart_data_type_composition(self, ctx: ChartContext, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_count = int(len(df.select_dtypes(include=["number"]).columns))
        categorical_count = int(len(df.columns) - numeric_count)
        total_columns = int(df.shape[1])
        rows = [
            {"type": "numeric", "count": numeric_count},
            {"type": "categorical", "count": categorical_count},
            {"type": "total", "count": total_columns},
        ]
        return self._save_chart(
            ctx,
            chart_id="data_type_composition",
            section="exploratory",
            title="Feature data type composition",
            purpose="Summarizes numeric and categorical coverage before diagnostics.",
            insight="Balanced feature types help diversify model signal and diagnostics.",
            chart_type="bar",
            payload={
                "xKey": "type",
                "rows": rows,
                "series": [{"key": "count", "name": "Count", "color": "#8ddcc2"}],
            },
        )

    def _chart_unique_value_density(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        features = [c for c in df.columns if c != target_column]
        if not features:
            return self._fallback_chart(
                ctx,
                chart_id="unique_value_density",
                section="exploratory",
                title="Unique value density by feature",
                purpose="Identifies near-constant and high-cardinality features.",
                insight="Fallback used because no predictor features are available.",
                fallback_payload={"rows": [{"label": "reason", "value": "no predictor features"}]},
            )

        sampled = features[:20]
        rows: List[Dict[str, Any]] = []
        total_rows = max(int(len(df)), 1)
        for col in sampled:
            nunique = int(df[col].nunique(dropna=True))
            density = float(nunique / total_rows)
            rows.append({
                "feature": col,
                "unique_values": nunique,
                "unique_ratio": round(density, 6),
            })

        return self._save_chart(
            ctx,
            chart_id="unique_value_density",
            section="exploratory",
            title="Unique value density by feature",
            purpose="Highlights low-variance and potentially high-cardinality columns.",
            insight="Very low ratios can indicate weak signal; very high ratios may imply sparse patterns.",
            chart_type="bar",
            payload={
                "xKey": "feature",
                "rows": rows,
                "series": [
                    {"key": "unique_values", "name": "Unique values", "color": "#6e3bd7"},
                    {"key": "unique_ratio", "name": "Unique ratio", "color": "#d0bcff"},
                ],
            },
        )

    def _chart_numeric_skewness_profile(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        numeric = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c != target_column]
        if not numeric:
            return self._fallback_chart(
                ctx,
                chart_id="numeric_skewness_profile",
                section="exploratory",
                title="Numeric skewness profile",
                purpose="Measures asymmetry of numeric feature distributions.",
                insight="Fallback used because no numeric feature exists.",
                fallback_payload={"rows": [{"label": "reason", "value": "no numeric columns"}]},
            )

        rows: List[Dict[str, Any]] = []
        for col in numeric[:15]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 3:
                continue
            rows.append({"feature": col, "skewness": float(s.skew())})

        if not rows:
            return self._fallback_chart(
                ctx,
                chart_id="numeric_skewness_profile",
                section="exploratory",
                title="Numeric skewness profile",
                purpose="Measures asymmetry of numeric feature distributions.",
                insight="Fallback used because there are not enough valid numeric rows.",
                fallback_payload={"rows": [{"label": "reason", "value": "insufficient numeric rows"}]},
            )

        rows = sorted(rows, key=lambda r: abs(r["skewness"]), reverse=True)
        return self._save_chart(
            ctx,
            chart_id="numeric_skewness_profile",
            section="exploratory",
            title="Numeric skewness profile",
            purpose="Highlights features with strong skew that can impact model fit.",
            insight="Highly skewed features may benefit from transforms like log or Box-Cox.",
            chart_type="bar",
            payload={
                "xKey": "feature",
                "rows": rows,
                "series": [{"key": "skewness", "name": "Skewness", "color": "#8ddcc2"}],
            },
        )

    def _chart_numeric_variance_profile(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        numeric = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c != target_column]
        if not numeric:
            return self._fallback_chart(
                ctx,
                chart_id="numeric_variance_profile",
                section="exploratory",
                title="Numeric variance profile",
                purpose="Compares spread of numeric predictors.",
                insight="Fallback used because no numeric feature exists.",
                fallback_payload={"rows": [{"label": "reason", "value": "no numeric columns"}]},
            )

        rows: List[Dict[str, Any]] = []
        for col in numeric[:15]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 2:
                continue
            rows.append({"feature": col, "variance": float(np.var(s.to_numpy(dtype=float)))})

        if not rows:
            return self._fallback_chart(
                ctx,
                chart_id="numeric_variance_profile",
                section="exploratory",
                title="Numeric variance profile",
                purpose="Compares spread of numeric predictors.",
                insight="Fallback used because there are not enough valid numeric rows.",
                fallback_payload={"rows": [{"label": "reason", "value": "insufficient numeric rows"}]},
            )

        rows = sorted(rows, key=lambda r: r["variance"], reverse=True)
        return self._save_chart(
            ctx,
            chart_id="numeric_variance_profile",
            section="exploratory",
            title="Numeric variance profile",
            purpose="Identifies low-variance and high-dispersion predictors.",
            insight="Near-zero variance features may add little predictive value.",
            chart_type="bar",
            payload={
                "xKey": "feature",
                "rows": rows,
                "series": [{"key": "variance", "name": "Variance", "color": "#d0bcff"}],
            },
        )

    def _chart_categorical_concentration(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        categorical = [
            c for c in df.columns
            if c != target_column and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if not categorical:
            return self._fallback_chart(
                ctx,
                chart_id="categorical_concentration",
                section="exploratory",
                title="Categorical concentration profile",
                purpose="Shows dominance of top categories across categorical features.",
                insight="Fallback used because no categorical feature exists.",
                fallback_payload={"rows": [{"label": "reason", "value": "no categorical columns"}]},
            )

        rows: List[Dict[str, Any]] = []
        for col in categorical[:12]:
            s = df[col].dropna().astype(str)
            total = max(len(s), 1)
            top_share = float(s.value_counts(normalize=True).iloc[0]) if len(s) > 0 else 0.0
            rows.append({
                "feature": col,
                "top_category_share": round(top_share, 6),
                "distinct_categories": int(s.nunique()),
                "non_null_rows": int(total),
            })

        return self._save_chart(
            ctx,
            chart_id="categorical_concentration",
            section="exploratory",
            title="Categorical concentration profile",
            purpose="Flags categorical columns dominated by a single level.",
            insight="High top-category share can signal weak categorical diversity.",
            chart_type="bar",
            payload={
                "xKey": "feature",
                "rows": rows,
                "series": [
                    {"key": "top_category_share", "name": "Top category share", "color": "#6e3bd7"},
                    {"key": "distinct_categories", "name": "Distinct categories", "color": "#8ddcc2"},
                ],
            },
        )

    def _chart_target_class_distribution(self, ctx: ChartContext, target: pd.Series) -> Dict[str, Any]:
        vc = target.astype(str).fillna("<missing>").value_counts().head(25)
        rows = [{"label": str(k), "count": int(v)} for k, v in vc.items()]
        return self._save_chart(
            ctx,
            chart_id="target_class_distribution",
            section="exploratory",
            title="Target class distribution",
            purpose="Checks class balance before model training.",
            insight="Strong imbalance may require stratification or class weighting.",
            chart_type="bar",
            payload={
                "xKey": "label",
                "rows": rows,
                "series": [{"key": "count", "name": "Count", "color": "#d0bcff"}],
            },
        )

    def _chart_target_distribution(self, ctx: ChartContext, target_numeric: pd.Series) -> Dict[str, Any]:
        s = target_numeric.dropna()
        counts, edges = np.histogram(s.to_numpy(dtype=float), bins=min(max(len(s) // 50, 8), 25)) if len(s) > 0 else ([], [])
        rows = []
        for i in range(len(counts)):
            rows.append({"label": f"{edges[i]:.3f}..{edges[i+1]:.3f}", "count": int(counts[i])})
        return self._save_chart(
            ctx,
            chart_id="target_distribution",
            section="exploratory",
            title="Target distribution",
            purpose="Shows response variable spread and skew.",
            insight="Heavy skew may benefit from transforms or robust models.",
            chart_type="bar",
            payload={
                "xKey": "label",
                "rows": rows,
                "series": [{"key": "count", "name": "Count", "color": "#6e3bd7"}],
            },
        )

    def _chart_missing_values(self, ctx: ChartContext, df: pd.DataFrame) -> Dict[str, Any]:
        rows = (
            df.isna().sum()
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
            .rename(columns={"index": "column", 0: "missing"})
            .to_dict(orient="records")
        )
        return self._save_chart(
            ctx,
            chart_id="missing_values_per_column",
            section="exploratory",
            title="Missing values per column",
            purpose="Identifies columns requiring imputation or removal.",
            insight="Columns with high missingness can degrade model stability.",
            chart_type="bar",
            payload={
                "xKey": "column",
                "rows": rows,
                "series": [{"key": "missing", "name": "Missing", "color": "#d0bcff"}],
            },
        )

    def _chart_numeric_distributions(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        numeric = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c != target_column]
        if not numeric:
            return self._fallback_chart(
                ctx,
                chart_id="numeric_feature_distributions",
                section="exploratory",
                title="Numeric feature distributions",
                purpose="Distribution diagnostics for numeric features.",
                insight="No numeric features available; fallback summary generated.",
                fallback_payload={"rows": [{"label": "numeric_features", "value": 0}]},
            )

        selected = numeric[:3]
        bins = 15
        rows: List[Dict[str, Any]] = []
        for col in selected:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            counts, edges = np.histogram(s.to_numpy(dtype=float), bins=bins)
            for i in range(len(counts)):
                rows.append({
                    "bin": f"{edges[i]:.3f}..{edges[i+1]:.3f}",
                    "feature": col,
                    "count": int(counts[i]),
                })

        return self._save_chart(
            ctx,
            chart_id="numeric_feature_distributions",
            section="exploratory",
            title="Numeric feature distributions",
            purpose="Highlights skewness and spread in key numeric features.",
            insight="Distribution anomalies often correlate with unstable training behavior.",
            chart_type="bar",
            payload={
                "xKey": "bin",
                "rows": rows,
                "series": [{"key": "count", "name": "Count", "color": "#6e3bd7"}],
                "groupKey": "feature",
            },
        )

    def _chart_boxplot_by_class(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        numeric = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c != target_column]
        if not numeric:
            return self._fallback_chart(
                ctx,
                chart_id="box_plots_by_class",
                section="exploratory",
                title="Box plots of numeric features by class",
                purpose="Compares class-conditioned feature spread.",
                insight="Fallback used because no numeric feature exists.",
                fallback_payload={"rows": [{"label": "reason", "value": "no numeric feature"}]},
            )

        feature = numeric[0]
        rows = []
        grp = df[[feature, target_column]].dropna().groupby(target_column)
        for cls, sub in grp:
            s = pd.to_numeric(sub[feature], errors="coerce").dropna()
            if len(s) == 0:
                continue
            rows.append(
                {
                    "class": str(cls),
                    "q1": float(s.quantile(0.25)),
                    "median": float(s.quantile(0.50)),
                    "q3": float(s.quantile(0.75)),
                }
            )

        return self._save_chart(
            ctx,
            chart_id="box_plots_by_class",
            section="exploratory",
            title="Box plots of numeric features by class",
            purpose="Shows class-wise central tendency and spread.",
            insight=f"Using '{feature}' as representative numeric feature.",
            chart_type="bar",
            payload={
                "xKey": "class",
                "rows": rows,
                "series": [
                    {"key": "q1", "name": "Q1", "color": "#d0bcff"},
                    {"key": "median", "name": "Median", "color": "#6e3bd7"},
                    {"key": "q3", "name": "Q3", "color": "#8ddcc2"},
                ],
            },
        )

    def _chart_correlation_heatmap(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        numeric = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c != target_column]
        if len(numeric) < 2:
            return self._fallback_chart(
                ctx,
                chart_id="correlation_heatmap",
                section="exploratory",
                title="Correlation heatmap",
                purpose="Shows linear correlation among numeric features.",
                insight="Fallback used because at least two numeric features are required.",
                fallback_payload={"rows": [{"label": "reason", "value": "insufficient numeric columns"}]},
            )

        selected = numeric[:8]
        corr = df[selected].corr(numeric_only=True).fillna(0.0)
        return self._save_chart(
            ctx,
            chart_id="correlation_heatmap",
            section="exploratory",
            title="Correlation heatmap",
            purpose="Highlights collinearity and redundancy.",
            insight="High absolute correlations may indicate redundant features.",
            chart_type="heatmap",
            payload={
                "xLabels": selected,
                "yLabels": selected,
                "matrix": [[float(v) for v in row] for row in corr.to_numpy().tolist()],
            },
        )

    def _chart_pairplot_top_numeric(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        numeric = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c != target_column]
        if len(numeric) < 2:
            return self._fallback_chart(
                ctx,
                chart_id="pairplot_top_numeric",
                section="exploratory",
                title="Pairplot for top numeric features",
                purpose="Visualizes pairwise relationships.",
                insight="Fallback used because at least two numeric features are required.",
                fallback_payload={"rows": [{"label": "reason", "value": "insufficient numeric columns"}]},
            )

        f1, f2 = numeric[0], numeric[1]
        sample = df[[f1, f2, target_column]].dropna().head(800)
        rows = [
            {"x": float(r[f1]), "y": float(r[f2]), "group": str(r[target_column])}
            for _, r in sample.iterrows()
            if np.isfinite(float(r[f1])) and np.isfinite(float(r[f2]))
        ]

        return self._save_chart(
            ctx,
            chart_id="pairplot_top_numeric",
            section="exploratory",
            title="Pairplot for top numeric features",
            purpose="Shows pairwise separation trend for leading numeric features.",
            insight=f"Rendered as scatter for '{f1}' vs '{f2}'.",
            chart_type="scatter",
            payload={
                "xKey": "x",
                "yKey": "y",
                "rows": rows,
            },
        )

    def _chart_categorical_vs_class_rate(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        categorical = [
            c for c in df.columns
            if c != target_column and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if not categorical:
            return self._fallback_chart(
                ctx,
                chart_id="categorical_vs_class_rate",
                section="exploratory",
                title="Categorical feature versus class rate comparison",
                purpose="Compares class behavior across categorical levels.",
                insight="Fallback used because no categorical feature exists.",
                fallback_payload={"rows": [{"label": "reason", "value": "no categorical columns"}]},
            )

        feature = categorical[0]
        tmp = df[[feature, target_column]].dropna().copy()
        tmp[feature] = tmp[feature].astype(str)
        top_values = tmp[feature].value_counts().head(8).index.tolist()
        tmp = tmp[tmp[feature].isin(top_values)]

        rows: List[Dict[str, Any]] = []
        for val, group in tmp.groupby(feature):
            total = max(len(group), 1)
            class_rates = group[target_column].astype(str).value_counts(normalize=True)
            row: Dict[str, Any] = {"category": str(val), "total": int(total)}
            for cls, rate in class_rates.items():
                row[f"rate_{cls}"] = float(rate)
            rows.append(row)

        series_keys = sorted({k for r in rows for k in r.keys() if k.startswith("rate_")})
        series = [{"key": key, "name": key.replace("rate_", ""), "color": "#d0bcff"} for key in series_keys]

        return self._save_chart(
            ctx,
            chart_id="categorical_vs_class_rate",
            section="exploratory",
            title="Categorical feature versus class rate comparison",
            purpose="Identifies categories associated with class concentration.",
            insight=f"Using '{feature}' as primary categorical feature.",
            chart_type="bar",
            payload={
                "xKey": "category",
                "rows": rows,
                "series": series,
            },
        )

    def _chart_feature_ranking(self, ctx: ChartContext, df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column]

        # Numeric encoding fallback for categoricals.
        X_enc = pd.get_dummies(X, drop_first=True)
        if X_enc.empty:
            return self._fallback_chart(
                ctx,
                chart_id="feature_ranking",
                section="exploratory",
                title="Feature importance / mutual information ranking",
                purpose="Ranks signal strength of features.",
                insight="Fallback used because no modelable feature remained after encoding.",
                fallback_payload={"rows": [{"label": "reason", "value": "no encoded features"}]},
            )

        # Trim to manageable width.
        X_enc = X_enc.iloc[:, :80]
        valid = X_enc.notna().all(axis=1) & y.notna()
        X_arr = X_enc.loc[valid]
        y_arr = y.loc[valid]
        if len(X_arr) < 5:
            return self._fallback_chart(
                ctx,
                chart_id="feature_ranking",
                section="exploratory",
                title="Feature importance / mutual information ranking",
                purpose="Ranks signal strength of features.",
                insight="Fallback used because too few valid rows are available.",
                fallback_payload={"rows": [{"label": "reason", "value": "insufficient valid rows"}]},
            )

        try:
            if task_type == "classification":
                scores = mutual_info_classif(X_arr, y_arr.astype(str), discrete_features=False, random_state=42)
            else:
                scores = mutual_info_regression(X_arr, pd.to_numeric(y_arr, errors="coerce").fillna(0.0), random_state=42)
            rows = [
                {"feature": col, "score": float(score)}
                for col, score in zip(X_arr.columns.tolist(), scores.tolist())
            ]
            rows = sorted(rows, key=lambda r: r["score"], reverse=True)[:15]
            return self._save_chart(
                ctx,
                chart_id="feature_ranking",
                section="exploratory",
                title="Feature importance / mutual information ranking",
                purpose="Ranks strongest predictive signals before model interpretation.",
                insight="Higher mutual information implies stronger nonlinear dependency with target.",
                chart_type="bar",
                payload={
                    "xKey": "feature",
                    "rows": rows,
                    "series": [{"key": "score", "name": "MI Score", "color": "#6e3bd7"}],
                },
            )
        except Exception as exc:
            return self._fallback_chart(
                ctx,
                chart_id="feature_ranking",
                section="exploratory",
                title="Feature importance / mutual information ranking",
                purpose="Ranks strongest predictive signals.",
                insight=f"Fallback used due to MI computation error: {exc}",
                fallback_payload={"rows": [{"label": "reason", "value": "mutual information failed"}]},
            )

    def _chart_outlier_analysis(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        numeric = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c != target_column]
        if not numeric:
            return self._fallback_chart(
                ctx,
                chart_id="outlier_analysis",
                section="exploratory",
                title="Outlier analysis for key numeric features",
                purpose="Detects extreme-value concentration using IQR bounds.",
                insight="Fallback used because no numeric feature exists.",
                fallback_payload={"rows": [{"label": "reason", "value": "no numeric columns"}]},
            )

        rows = []
        for col in numeric[:12]:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 5:
                continue
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                outliers = 0
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = int(((s < lower) | (s > upper)).sum())
            rows.append({"feature": col, "outliers": outliers})

        return self._save_chart(
            ctx,
            chart_id="outlier_analysis",
            section="exploratory",
            title="Outlier analysis for key numeric features",
            purpose="Flags potential instability due to extreme values.",
            insight="Features with high outlier counts may need clipping or robust scaling.",
            chart_type="bar",
            payload={
                "xKey": "feature",
                "rows": rows,
                "series": [{"key": "outliers", "name": "Outliers", "color": "#d0bcff"}],
            },
        )

    def _chart_feature_vs_target_scatter(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        numeric = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c != target_column]
        if not numeric:
            return self._fallback_chart(
                ctx,
                chart_id="feature_vs_target_scatter",
                section="exploratory",
                title="Scatter plots of key features versus target",
                purpose="Shows monotonic and nonlinear associations with target.",
                insight="Fallback used because no numeric feature exists.",
                fallback_payload={"rows": [{"label": "reason", "value": "no numeric columns"}]},
            )

        feature = numeric[0]
        sample = df[[feature, target_column]].dropna().head(1200)
        rows = [{"x": float(r[feature]), "y": float(r[target_column])} for _, r in sample.iterrows()]
        return self._save_chart(
            ctx,
            chart_id="feature_vs_target_scatter",
            section="exploratory",
            title="Scatter plots of key features versus target",
            purpose="Reveals slope, heteroscedasticity, and nonlinear response behavior.",
            insight=f"Using '{feature}' as representative predictor.",
            chart_type="scatter",
            payload={"xKey": "x", "yKey": "y", "rows": rows},
        )

    def _chart_binned_target_by_category(self, ctx: ChartContext, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        categorical = [
            c for c in df.columns
            if c != target_column and not pd.api.types.is_numeric_dtype(df[c])
        ]
        if not categorical:
            return self._fallback_chart(
                ctx,
                chart_id="binned_target_analysis",
                section="exploratory",
                title="Binned target analysis for categorical features",
                purpose="Compares mean target values across category bins.",
                insight="Fallback used because no categorical feature exists.",
                fallback_payload={"rows": [{"label": "reason", "value": "no categorical columns"}]},
            )

        feature = categorical[0]
        tmp = df[[feature, target_column]].dropna().copy()
        tmp[feature] = tmp[feature].astype(str)
        top_values = tmp[feature].value_counts().head(12).index.tolist()
        tmp = tmp[tmp[feature].isin(top_values)]
        rows = (
            tmp.groupby(feature)[target_column]
            .mean()
            .reset_index()
            .rename(columns={feature: "category", target_column: "target_mean"})
            .to_dict(orient="records")
        )
        return self._save_chart(
            ctx,
            chart_id="binned_target_analysis",
            section="exploratory",
            title="Binned target analysis for important categorical features",
            purpose="Highlights target drift across category buckets.",
            insight=f"Using '{feature}' categories with highest support.",
            chart_type="bar",
            payload={
                "xKey": "category",
                "rows": rows,
                "series": [{"key": "target_mean", "name": "Mean target", "color": "#6e3bd7"}],
            },
        )

    def _chart_residual_distribution(self, ctx: ChartContext, residuals: np.ndarray) -> Dict[str, Any]:
        counts, edges = np.histogram(residuals, bins=min(max(len(residuals) // 50, 10), 25)) if len(residuals) > 0 else ([], [])
        rows = [
            {"label": f"{edges[i]:.3f}..{edges[i+1]:.3f}", "count": int(counts[i])}
            for i in range(len(counts))
        ]
        return self._save_chart(
            ctx,
            chart_id="residual_distribution",
            section="model_evaluation",
            title="Residual distribution",
            purpose="Checks error centering and heavy-tail behavior.",
            insight="Residuals should center near zero with limited heavy tails.",
            chart_type="bar",
            payload={
                "xKey": "label",
                "rows": rows,
                "series": [{"key": "count", "name": "Count", "color": "#d0bcff"}],
            },
        )

    def _chart_residuals_vs_predicted(self, ctx: ChartContext, y_pred: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        rows = [{"x": float(p), "y": float(r)} for p, r in zip(y_pred.tolist(), residuals.tolist())][:2000]
        return self._save_chart(
            ctx,
            chart_id="residuals_vs_predicted",
            section="model_evaluation",
            title="Residuals versus predicted values",
            purpose="Detects systematic bias and heteroscedasticity.",
            insight="Patterns or funnel shapes indicate model misspecification.",
            chart_type="scatter",
            payload={"xKey": "x", "yKey": "y", "rows": rows},
        )

    def _chart_actual_vs_predicted(self, ctx: ChartContext, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        rows = [{"x": float(t), "y": float(p)} for t, p in zip(y_true.tolist(), y_pred.tolist())][:2000]
        return self._save_chart(
            ctx,
            chart_id="actual_vs_predicted_parity",
            section="model_evaluation",
            title="Actual versus predicted scatter (parity)",
            purpose="Measures agreement between predictions and ground truth.",
            insight="Points closer to diagonal indicate stronger fit.",
            chart_type="scatter",
            payload={"xKey": "x", "yKey": "y", "rows": rows},
        )

    # ---- Artifact + fallback -----------------------------------------------------

    def _save_chart(
        self,
        ctx: ChartContext,
        *,
        chart_id: str,
        section: str,
        title: str,
        purpose: str,
        insight: str,
        chart_type: str,
        payload: Dict[str, Any],
        fallback_used: bool = False,
    ) -> Dict[str, Any]:
        artifact_dir = os.path.join(ctx.reports_root, str(ctx.dataset_id), ctx.report_id)
        os.makedirs(artifact_dir, exist_ok=True)

        safe_chart_id = chart_id.replace(" ", "_").lower()
        filename = f"{safe_chart_id}.json"
        file_path = os.path.join(artifact_dir, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        rel = os.path.relpath(file_path, self.reports_root).replace("\\", "/")
        return {
            "id": safe_chart_id,
            "section": section,
            "title": title,
            "description": purpose,
            "purpose": purpose,
            "insight": insight,
            "chart_type": chart_type,
            "fallback_used": fallback_used,
            "output_reference": {
                "path": file_path,
                "relative_path": rel,
                "url": f"/static/reports/{rel}",
            },
            "artifact_ref": {
                "path": file_path,
                "relative_path": rel,
                "url": f"/static/reports/{rel}",
            },
            "spec": payload,
        }

    def _fallback_chart(
        self,
        ctx: ChartContext,
        *,
        chart_id: str,
        section: str,
        title: str,
        purpose: str,
        insight: str,
        fallback_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._save_chart(
            ctx,
            chart_id=chart_id,
            section=section,
            title=title,
            purpose=purpose,
            insight=insight,
            chart_type="table",
            payload=fallback_payload,
            fallback_used=True,
        )

    def _padding_charts(self, ctx: ChartContext, missing: int) -> List[Dict[str, Any]]:
        pads: List[Dict[str, Any]] = []
        for i in range(missing):
            pads.append(
                self._fallback_chart(
                    ctx,
                    chart_id=f"fallback_padding_{i+1}",
                    section="exploratory",
                    title=f"Fallback diagnostic {i+1}",
                    purpose="Ensures minimum chart count guarantee is enforced.",
                    insight="Generated as fallback to satisfy minimum chart coverage.",
                    fallback_payload={"rows": [{"label": "generated", "value": True}]},
                )
            )
        return pads


def build_predictions_from_model_bundle(
    df: pd.DataFrame,
    *,
    target_column: str,
    model_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Build y_true/y_pred payload for evaluation from a saved model bundle."""
    pipeline = model_bundle.get("pipeline")
    if pipeline is None:
        raise ValueError("Model bundle missing pipeline")

    feature_names: Sequence[str] = model_bundle.get("feature_names") or []
    if not feature_names:
        feature_names = [c for c in df.columns if c != target_column]

    if target_column not in df.columns:
        raise ValueError("Target column missing from dataset")

    frame = df.copy()
    for col in feature_names:
        if col not in frame.columns:
            frame[col] = np.nan

    X = frame[list(feature_names)]
    y_true = frame[target_column]

    valid = y_true.notna()
    X = X.loc[valid]
    y_true = y_true.loc[valid]

    y_pred = pipeline.predict(X)

    return {
        "y_true": y_true.to_numpy().tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
    }
