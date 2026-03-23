import numpy as np
import pandas as pd

from app.ml.analytics_service import AnalyticsReportService


def _build_classification_df(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, rows)
    x2 = rng.normal(1, 2, rows)
    cat = np.where(rng.random(rows) > 0.5, "A", "B")
    target = np.where(x1 + 0.4 * x2 + (cat == "A").astype(int) > 0.8, "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "target": target})


def _build_regression_df(rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x1 = rng.normal(0, 1, rows)
    x2 = rng.normal(2, 1.5, rows)
    cat = np.where(rng.random(rows) > 0.6, "east", "west")
    y = 3.2 * x1 - 1.1 * x2 + (cat == "east").astype(float) * 0.7 + rng.normal(0, 0.5, rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "region": cat, "target": y})


def _assert_chart_contract(charts):
    for c in charts:
        assert c.get("title")
        assert c.get("description")
        assert c.get("purpose")
        assert c.get("insight")
        assert c.get("artifact_ref")
        assert c.get("output_reference")


def test_minimum_10_charts_returned() -> None:
    svc = AnalyticsReportService()
    df = _build_classification_df()

    report = svc.generate_report(
        dataset_id=1,
        dataset_name="cls_ds",
        df=df,
        target_column="target",
    )

    assert report["chart_count"] >= 10
    _assert_chart_contract(report["exploratory_charts"] + report["evaluation_charts"])


def test_minimum_10_charts_returned_for_regression() -> None:
    svc = AnalyticsReportService()
    df = _build_regression_df()

    report = svc.generate_report(
        dataset_id=11,
        dataset_name="reg_min_ds",
        df=df,
        target_column="target",
    )

    assert report["chart_count"] >= 10
    _assert_chart_contract(report["exploratory_charts"] + report["evaluation_charts"])


def test_generates_more_than_minimum_when_dataset_supports_extra_diagnostics() -> None:
    svc = AnalyticsReportService()
    df = _build_classification_df(rows=400)

    report = svc.generate_report(
        dataset_id=12,
        dataset_name="extra_cls_ds",
        df=df,
        target_column="target",
    )

    assert report["minimum_required_charts"] == 10
    assert report["chart_count"] > report["minimum_required_charts"]


def test_regression_includes_parity_and_residual_when_predictions_exist() -> None:
    svc = AnalyticsReportService()
    df = _build_regression_df()

    y_true = df["target"].to_numpy()
    y_pred = y_true + np.random.default_rng(99).normal(0, 0.25, len(y_true))

    report = svc.generate_report(
        dataset_id=2,
        dataset_name="reg_ds",
        df=df,
        target_column="target",
        predictions={"y_true": y_true.tolist(), "y_pred": y_pred.tolist()},
    )

    eval_ids = {c["id"] for c in report["evaluation_charts"]}
    assert "actual_vs_predicted_parity" in eval_ids
    assert "residual_distribution" in eval_ids
    assert "residuals_vs_predicted" in eval_ids


def test_classification_includes_class_level_eval_when_predictions_exist() -> None:
    svc = AnalyticsReportService()
    df = _build_classification_df()

    y_true = df["target"].to_numpy()
    y_pred = y_true.copy()
    y_pred[:15] = np.where(y_pred[:15] == "yes", "no", "yes")

    report = svc.generate_report(
        dataset_id=3,
        dataset_name="cls_eval_ds",
        df=df,
        target_column="target",
        predictions={"y_true": y_true.tolist(), "y_pred": y_pred.tolist()},
    )

    eval_ids = {c["id"] for c in report["evaluation_charts"]}
    assert "classification_confusion_matrix" in eval_ids
    assert "classification_per_class_metrics" in eval_ids


def test_every_chart_has_required_metadata_fields() -> None:
    svc = AnalyticsReportService()
    df = _build_regression_df()

    report = svc.generate_report(
        dataset_id=4,
        dataset_name="meta_ds",
        df=df,
        target_column="target",
    )

    charts = report["exploratory_charts"] + report["evaluation_charts"]
    _assert_chart_contract(charts)
