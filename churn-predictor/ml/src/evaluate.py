"""
evaluate.py
===========
Load trained pipeline and held-out test set → compute all metrics → write metrics.json.

Usage:
    python evaluate.py [--data path/to/data.csv]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocess import MODELS_DIR, _find_csv, preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

METRICS_PATH = MODELS_DIR / "metrics.json"
MODEL_VERSION = "1.0.0"


def evaluate(csv_path: Path) -> dict:
    """Run full evaluation on held-out test set. Returns metrics dict."""

    # ---- Load artifacts ----
    pipeline_path = MODELS_DIR / "churn_pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"No trained model found at {pipeline_path}. Run train.py first.")

    artifact = joblib.load(pipeline_path)
    preprocessor = artifact["preprocessor"]
    xgb_model = artifact["xgb_model"]

    log.info("Model loaded from %s", pipeline_path)

    # ---- Re-run preprocessing (same random_state ensures same split) ----
    X_train, X_test, y_train, y_test, _, feature_names = preprocess(
        csv_path, save_preprocessor=False
    )

    # ---- Inference ----
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # ---- Metrics ----
    roc_auc = float(roc_auc_score(y_test, y_prob))
    avg_precision = float(average_precision_score(y_test, y_prob))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    # ---- Log to console ----
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    log.info("=" * 50)
    log.info("TEST SET RESULTS")
    log.info("  ROC-AUC          : %.4f", roc_auc)
    log.info("  Avg Precision    : %.4f", avg_precision)
    log.info("  Precision        : %.4f", precision)
    log.info("  Recall           : %.4f", recall)
    log.info("  F1               : %.4f", f1)
    log.info("  Confusion Matrix : TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)
    log.info("=" * 50)

    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_samples": int(len(y_test)),
        "churn_rate_test": float(y_test.mean()),
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    # ---- Write metrics.json ----
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics written → %s", METRICS_PATH)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()

    csv_path = args.data or _find_csv()
    metrics = evaluate(csv_path)
    print(f"\n[OK] Test ROC-AUC: {metrics['roc_auc']:.4f}")
