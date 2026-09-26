"""
model_service.py
================
Loads the trained joblib pipeline once at startup and exposes:
  - predict_single(customer_input) → PredictionResponse
  - predict_batch(df) → list[BatchPredictionRow]
  - get_metadata() → ModelMetadata
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

log = logging.getLogger(__name__)

# Paths — resolved relative to this file's location so they work
# both in development (api/app/services/) and in Docker (/app/app/services/)
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]          # churn-predictor/
_ML_MODELS = _REPO_ROOT / "ml" / "models"

PIPELINE_PATH = _ML_MODELS / "churn_pipeline.joblib"
METRICS_PATH = _ML_MODELS / "metrics.json"

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_30d",
    "contract_type",
    "payment_method",
]


def _risk_tier(prob: float) -> str:
    if prob >= 0.65:
        return "high"
    elif prob >= 0.35:
        return "medium"
    return "low"


class ModelService:
    """Singleton that holds the loaded model + explainer."""

    _instance: "ModelService | None" = None

    def __init__(self) -> None:
        log.info("Loading model artifact from %s", PIPELINE_PATH)
        if not PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {PIPELINE_PATH}. "
                "Run ml/src/train.py and ml/src/evaluate.py first."
            )

        artifact: dict[str, Any] = joblib.load(PIPELINE_PATH)
        self.preprocessor = artifact["preprocessor"]
        self.xgb_model = artifact["xgb_model"]
        self.feature_names: list[str] = artifact["feature_names"]

        log.info("Building SHAP TreeExplainer…")
        self.explainer = shap.TreeExplainer(self.xgb_model)

        log.info("Model loaded. Features: %d", len(self.feature_names))

    @classmethod
    def get(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------

    def _df_from_customer(self, data: dict) -> pd.DataFrame:
        row = {k: [v] for k, v in data.items() if k in FEATURE_COLUMNS}
        return pd.DataFrame(row)[FEATURE_COLUMNS]

    def _shap_factors(self, X_transformed: np.ndarray, n: int = 3) -> list[dict]:
        sv = self.explainer.shap_values(X_transformed)
        if isinstance(sv, list):
            sv = sv[1]  # class-1 SHAP for binary classifiers
        sv = sv[0]  # first (only) row

        indexed = sorted(enumerate(sv), key=lambda x: abs(x[1]), reverse=True)[:n]
        factors = []
        for i, shap_val in indexed:
            fname = self.feature_names[i] if i < len(self.feature_names) else f"feat_{i}"
            fval = float(X_transformed[0, i])
            factors.append({
                "feature": fname,
                "value": round(fval, 4) if np.isfinite(fval) else None,
                "shap_impact": round(float(shap_val), 6),
                "direction": "increases_churn" if shap_val > 0 else "decreases_churn",
            })
        return factors

    def predict_single(self, customer: dict) -> dict:
        df = self._df_from_customer(customer)
        X = self.preprocessor.transform(df)
        prob = float(self.xgb_model.predict_proba(X)[0, 1])
        factors = self._shap_factors(X)
        return {
            "customer_id": customer.get("customer_id"),
            "churn_probability": round(prob, 4),
            "risk_tier": _risk_tier(prob),
            "top_factors": factors,
        }

    def predict_batch(self, df: pd.DataFrame) -> list[dict]:
        # Align columns
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Missing column in batch input: {col}")

        X = self.preprocessor.transform(df[FEATURE_COLUMNS])
        probs = self.xgb_model.predict_proba(X)[:, 1]
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]

        results = []
        for i, (prob, sv_row) in enumerate(zip(probs, sv)):
            # Top-3 SHAP for this row
            indexed = sorted(enumerate(sv_row), key=lambda x: abs(x[1]), reverse=True)[:3]
            factors = []
            for fi, fval in indexed:
                fname = self.feature_names[fi] if fi < len(self.feature_names) else f"feat_{fi}"
                factors.append({
                    "feature": fname,
                    "value": round(float(X[i, fi]), 4),
                    "shap_impact": round(float(fval), 6),
                    "direction": "increases_churn" if fval > 0 else "decreases_churn",
                })

            cid = str(df.iloc[i].get("customer_id", "")) or None
            results.append({
                "row_index": i,
                "customer_id": cid,
                "churn_probability": round(float(prob), 4),
                "risk_tier": _risk_tier(float(prob)),
                "top_factors": factors,
            })

        return results

    def get_metadata(self) -> dict:
        if not METRICS_PATH.exists():
            return {"error": "metrics.json not found — run evaluate.py"}
        with open(METRICS_PATH) as f:
            return json.load(f)
