"""
train.py
========
1. Baseline LogisticRegression
2. XGBClassifier tuned with Optuna (≥20 trials, ROC-AUC objective)
3. SMOTE applied inside imblearn Pipeline (train fold only)
4. All runs logged to MLflow
5. Best pipeline persisted → ml/models/churn_pipeline.joblib

Usage:
    python train.py [--data path/to/data.csv] [--trials 25]
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# Silence pandera import deprecation (pandera < 0.20 shows this on top-level import)
os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from preprocess import MODELS_DIR, _find_csv, build_preprocessor, preprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = ROOT / "mlruns"
MLRUNS_DIR.mkdir(exist_ok=True)

# MLflow 3.x requires a database backend (file store is deprecated)
MLFLOW_DB = ROOT / "mlruns" / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
EXPERIMENT_NAME = "churn-prediction"
mlflow.set_experiment(EXPERIMENT_NAME)

CV_FOLDS = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Baseline — Logistic Regression
# ---------------------------------------------------------------------------

def run_baseline(X_train: np.ndarray, y_train: np.ndarray) -> float:
    log.info("Running baseline LogisticRegression…")
    with mlflow.start_run(run_name="baseline-logreg"):
        lr = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )
        scores = cross_val_score(
            lr, X_train, y_train,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="roc_auc", n_jobs=-1,
        )
        auc = float(scores.mean())
        mlflow.log_params({"model": "LogisticRegression", "max_iter": 1000, "class_weight": "balanced"})
        mlflow.log_metric("cv_roc_auc_mean", auc)
        mlflow.log_metric("cv_roc_auc_std", float(scores.std()))
        log.info("Baseline ROC-AUC: %.4f ± %.4f", auc, scores.std())
    return auc


# ---------------------------------------------------------------------------
# Optuna Objective — XGBoost + SMOTE
# ---------------------------------------------------------------------------

def make_xgb_pipeline(params: dict) -> ImbPipeline:
    """Wrap SMOTE + XGBClassifier in an imblearn Pipeline."""
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "model",
                XGBClassifier(
                    **params,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0,
                ),
            ),
        ]
    )


def objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
    }

    pipeline = make_xgb_pipeline(params)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    return float(scores.mean())


def run_optuna_tuning(
    X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 25
) -> tuple[dict, float]:
    log.info("Starting Optuna hyperparameter search (%d trials)…", n_trials)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_auc = study.best_value
    log.info("Best trial ROC-AUC: %.4f | params: %s", best_auc, best_params)

    # Log summary of all trials as a single MLflow run
    with mlflow.start_run(run_name="optuna-all-trials"):
        for trial in study.trials:
            mlflow.log_metric(f"trial_{trial.number}_auc", trial.value or 0.0)
        mlflow.log_metric("best_cv_roc_auc", best_auc)
        mlflow.log_param("n_trials", len(study.trials))

    return best_params, best_auc



# ---------------------------------------------------------------------------
# Final pipeline assembly
# ---------------------------------------------------------------------------

def build_final_pipeline(
    best_xgb_params: dict,
    preprocessor,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ImbPipeline:
    """
    Build the production pipeline:
        preprocessor → SMOTE → XGBClassifier
    Fit on the full training set.
    """
    # NOTE: preprocessor is already fitted; we wrap it as a passthrough
    # by using the already-transformed X_train.  The persisted artifact
    # stores the preprocessor separately so the API can also transform
    # single records via preprocessor.transform().
    #
    # For simplicity in serving, we also store a combined joblib that
    # chains: raw_preprocessor → smote(train-only) → xgb.
    # At inference time only preprocessor + xgb steps are used (no SMOTE).

    xgb = XGBClassifier(
        **best_xgb_params,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    log.info("Fitting final XGBClassifier on full training set…")

    # Apply SMOTE once on the full training data
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    log.info(
        "After SMOTE: %d samples | churn=%.2f%%",
        len(y_resampled), y_resampled.mean() * 100,
    )

    xgb.fit(X_resampled, y_resampled)

    # The inference pipeline: preprocessor → xgb (no SMOTE at inference)
    inference_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", xgb),
        ]
    )
    # Fit is a no-op since both steps are already fitted; calling anyway
    # so the pipeline state is consistent.
    inference_pipeline.steps[0][1].__setattr__("_is_fitted", True)  # mark fitted

    return inference_pipeline, xgb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(csv_path: Path, n_trials: int = 25) -> str:
    """Full training run. Returns path to saved pipeline."""

    # ---- Preprocess ----
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess(
        csv_path, save_preprocessor=True
    )

    # ---- Baseline ----
    run_baseline(X_train, y_train)

    # ---- Optuna ----
    best_params, best_cv_auc = run_optuna_tuning(X_train, y_train, n_trials=n_trials)


    # ---- Final fit ----
    with mlflow.start_run(run_name="final-model"):
        inference_pipeline, xgb_model = build_final_pipeline(
            best_params, preprocessor, X_train, y_train
        )

        # Quick train-set AUC
        train_probs = xgb_model.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train, train_probs)
        mlflow.log_metric("train_roc_auc", train_auc)
        mlflow.log_params(best_params)
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("feature_count", str(X_train.shape[1]))

        # Save artifacts
        pipeline_path = MODELS_DIR / "churn_pipeline.joblib"
        artifact = {
            "pipeline": inference_pipeline,
            "preprocessor": preprocessor,
            "xgb_model": xgb_model,
            "feature_names": feature_names,
            "best_params": best_params,
        }
        joblib.dump(artifact, pipeline_path)
        mlflow.log_artifact(str(pipeline_path))
        log.info("Pipeline saved → %s", pipeline_path)

    log.info("Training complete. Train ROC-AUC: %.4f", train_auc)
    return str(pipeline_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()

    csv_path = args.data or _find_csv()
    train(csv_path, n_trials=args.trials)
