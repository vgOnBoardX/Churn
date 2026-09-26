"""
explain.py
==========
SHAP-based model explainability:
  - Global: SHAP summary plot + feature_importance.json (top-15 drivers)
  - Per-customer: helper function used by the API to get top-3 factors per prediction

Usage:
    python explain.py [--data path/to/data.csv] [--sample 500]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import shap

from preprocess import MODELS_DIR, _find_csv, preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.json"
SHAP_PLOT_PATH = MODELS_DIR / "shap_summary.png"
SHAP_VALUES_PATH = MODELS_DIR / "shap_values_sample.joblib"


def compute_shap(
    csv_path: Path, n_sample: int = 500
) -> tuple[np.ndarray, np.ndarray, list[str], shap.TreeExplainer]:
    """
    Compute SHAP values on a sample of the test set.

    Returns (shap_values, X_sample, feature_names, explainer)
    """
    artifact = joblib.load(MODELS_DIR / "churn_pipeline.joblib")
    preprocessor = artifact["preprocessor"]
    xgb_model = artifact["xgb_model"]
    feature_names: list[str] = artifact["feature_names"]

    # Load test set (same random state → reproducible)
    _, X_test, _, y_test, _, _ = preprocess(csv_path, save_preprocessor=False)

    # Sample for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_test), size=min(n_sample, len(X_test)), replace=False)
    X_sample = X_test[idx]

    log.info("Computing SHAP values for %d samples…", len(X_sample))
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    return shap_values, X_sample, feature_names, explainer


def global_feature_importance(shap_values: np.ndarray, feature_names: list[str]) -> list[dict]:
    """Compute mean |SHAP| per feature → sorted list."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    pairs = sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"feature": f, "importance": round(v, 6)} for f, v in pairs]


def save_shap_plot(shap_values: np.ndarray, X_sample: np.ndarray, feature_names: list[str]) -> None:
    log.info("Generating SHAP summary plot…")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("SHAP plot saved → %s", SHAP_PLOT_PATH)


def top_shap_factors_for_instance(
    shap_instance: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray,
    n: int = 3,
) -> list[dict]:
    """
    Extract top N SHAP factors for a single prediction instance.

    Returns list of dicts:
        {feature, value, shap_impact, direction}
    """
    indexed = sorted(
        enumerate(shap_instance),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:n]

    factors = []
    for i, shap_val in indexed:
        feat_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        factors.append(
            {
                "feature": feat_name,
                "value": round(float(feature_values[i]), 4) if np.isfinite(feature_values[i]) else None,
                "shap_impact": round(float(shap_val), 6),
                "direction": "increases_churn" if shap_val > 0 else "decreases_churn",
            }
        )
    return factors


def explain(csv_path: Path, n_sample: int = 500) -> list[dict]:
    """Run full explainability pipeline. Returns global feature importance list."""
    shap_values, X_sample, feature_names, explainer = compute_shap(csv_path, n_sample)

    # Global importance
    importance = global_feature_importance(shap_values, feature_names)
    top_15 = importance[:15]

    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump({"top_features": top_15}, f, indent=2)
    log.info("Feature importance written → %s", FEATURE_IMPORTANCE_PATH)

    # Summary plot
    save_shap_plot(shap_values, X_sample, feature_names)

    # Save SHAP sample values for API warm-up (optional cache)
    joblib.dump(
        {"shap_values": shap_values, "X_sample": X_sample, "feature_names": feature_names},
        SHAP_VALUES_PATH,
    )
    log.info("SHAP sample values saved → %s", SHAP_VALUES_PATH)

    log.info("\nTop 10 global churn drivers:")
    for i, item in enumerate(importance[:10], 1):
        log.info("  %2d. %-35s %.6f", i, item["feature"], item["importance"])

    return importance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=500)
    args = parser.parse_args()

    csv_path = args.data or _find_csv()
    explain(csv_path, n_sample=args.sample)
