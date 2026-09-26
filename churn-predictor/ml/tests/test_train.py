import pytest
from pathlib import Path
import joblib
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PIPELINE_PATH = MODELS_DIR / "churn_pipeline.joblib"


def test_pipeline_artifact_exists():
    assert PIPELINE_PATH.exists(), f"Expected pipeline artifact at {PIPELINE_PATH}"


def test_pipeline_artifact_contents():
    artifact = joblib.load(PIPELINE_PATH)
    assert "preprocessor" in artifact
    assert "xgb_model" in artifact
    assert "feature_names" in artifact
    assert len(artifact["feature_names"]) > 0


def test_model_inference_shape():
    artifact = joblib.load(PIPELINE_PATH)
    preprocessor = artifact["preprocessor"]
    xgb_model = artifact["xgb_model"]

    import pandas as pd

    sample_df = pd.DataFrame(
        [
            {
                "tenure_months": 12,
                "monthly_charges": 65.5,
                "total_charges": 786.0,
                "contract_type": "month-to-month",
                "payment_method": "electronic_check",
                "support_tickets_30d": 2,
            }
        ]
    )

    X_transformed = preprocessor.transform(sample_df)
    probs = xgb_model.predict_proba(X_transformed)

    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0
