import os
import tempfile
from types import SimpleNamespace

import pandas as pd
import pytest

from app.ml.explainability import ExplainabilityError, explainability_service


def test_explainability_context_missing_files_raises():
    model = SimpleNamespace(
        id=1,
        model_name="m",
        model_type="RandomForestClassifier",
        task_type="classification",
        job_id="job-1",
        file_path="/tmp/does-not-exist.joblib",
    )

    with pytest.raises(ExplainabilityError) as exc:
        explainability_service.build_context(model, "/tmp/also-missing.csv")

    assert exc.value.code == "MODEL_FILE_MISSING"


def test_prepare_dataset_validates_target_column(tmp_path: tempfile.TemporaryDirectory):
    csv_path = os.path.join(tmp_path, "data.csv")
    pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(csv_path, index=False)

    with pytest.raises(ExplainabilityError) as exc:
        explainability_service._prepare_dataset(csv_path, "target")

    assert exc.value.code == "TARGET_COLUMN_MISSING"
