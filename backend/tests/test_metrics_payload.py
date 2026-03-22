from types import SimpleNamespace

from app.ml.metrics_payload import build_metrics_payload


def _model(**kwargs):
    defaults = dict(
        id=1,
        model_name="Random Forest",
        model_type="RandomForestClassifier",
        task_type="classification",
        training_time=1.2,
        is_deployed=False,
        metrics={"accuracy": 0.91, "f1_score": 0.9},
        cv_scores=[0.9, 0.92],
        feature_importance={"x": 0.5},
        confusion_matrix=[[8, 1], [2, 9]],
        roc_curve_data={"fpr": [0.0, 0.2, 1.0], "tpr": [0.0, 0.8, 1.0], "thresholds": [1.5, 0.5, 0.0]},
        accuracy=0.91,
        f1_score=0.9,
        roc_auc=0.95,
        rmse=None,
        mae=None,
        r2_score=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_metrics_payload_contains_chart_ready_sections():
    payload = build_metrics_payload(
        job_id="job-1",
        task_type="classification",
        status="completed",
        models=[_model()],
    )

    assert payload["job_id"] == "job-1"
    assert payload["metric_catalog"]
    assert payload["chart_data"]["metric_comparison"]["labels"] == ["Random Forest"]
    assert payload["chart_data"]["roc_curves"][0]["curve"]["fpr"][1] == 0.2
    assert payload["chart_data"]["confusion_matrices"][0]["matrix"]["matrix"][0][0] == 8
