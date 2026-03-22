import pandas as pd

from app.ml.dataset_profiling import build_dataset_profile


def test_dataset_profile_builds_compact_payload():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, None, 4.0],
            "b": ["x", "y", "x", None],
            "target": [0, 1, 0, 1],
        }
    )

    payload = build_dataset_profile(df, dataset_id=10, dataset_name="demo", target_column="target")

    assert payload["dataset_id"] == 10
    assert payload["total_rows"] == 4
    assert payload["total_columns"] == 3
    assert payload["missing_total"] == 2
    assert isinstance(payload["columns"], list)
    assert payload["target_distribution"] is not None
    assert "summary_cards" in payload
