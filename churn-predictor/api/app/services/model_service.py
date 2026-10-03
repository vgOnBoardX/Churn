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
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger(__name__)

# Paths — resolved flexibly across development, Docker, and Vercel environments
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]          # churn-predictor/


def _resolve_path(env_var: str, fallback_candidates: list[Path]) -> Path:
    env_val = os.getenv(env_var)
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
    for p in fallback_candidates:
        if p.exists():
            return p
    return fallback_candidates[0]


PIPELINE_PATH = _resolve_path("MODEL_PATH", [
    _REPO_ROOT / "ml" / "models" / "churn_pipeline.joblib",
    Path("churn-predictor/ml/models/churn_pipeline.joblib"),
    Path("ml/models/churn_pipeline.joblib"),
    _HERE.parents[2] / "ml" / "models" / "churn_pipeline.joblib",
])

METRICS_PATH = _resolve_path("METRICS_PATH", [
    _REPO_ROOT / "ml" / "models" / "metrics.json",
    Path("churn-predictor/ml/models/metrics.json"),
    Path("ml/models/metrics.json"),
    _HERE.parents[2] / "ml" / "models" / "metrics.json",
])

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
    """Singleton that holds the loaded model + native Tree SHAP explainer."""

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

        # Optional SHAP library fallback (XGBoost computes exact Tree SHAP via pred_contribs=True)
        self.explainer = None
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.xgb_model)
            log.info("SHAP TreeExplainer initialized as fallback.")
        except ImportError:
            log.info("Using native XGBoost Tree SHAP calculation (zero extra dependencies).")

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

    def _compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Tree SHAP feature attributions.
        Uses XGBoost native C++ Lundberg & Lee Tree SHAP implementation (pred_contribs=True),
        falling back to shap package if available.
        """
        try:
            dmat = xgb.DMatrix(X)
            contribs = self.xgb_model.get_booster().predict(dmat, pred_contribs=True)
            # contribs shape: (n_samples, n_features + 1), where last column is the base value / bias
            return contribs[:, :-1]
        except Exception as exc:
            if self.explainer is not None:
                sv = self.explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]
                return np.array(sv)
            log.exception("Native SHAP calculation failed: %s", exc)
            raise

    def _shap_factors(self, X_transformed: np.ndarray, n: int = 3) -> list[dict]:
        sv_all = self._compute_shap_values(X_transformed)
        sv = sv_all[0]  # first (only) row

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
        sv = self._compute_shap_values(X)

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
