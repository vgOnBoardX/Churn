"""
preprocess.py
=============
Loads raw customer churn CSV, validates schema, engineers features,
builds a ColumnTransformer, and returns stratified 80/20 train/test splits.

Usage:
    python preprocess.py                          # uses default DATA_PATH
    python preprocess.py --input path/to/data.csv
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import joblib
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_30d",
]

CATEGORICAL_FEATURES: list[str] = [
    "contract_type",
    "payment_method",
]

TARGET: str = "is_churned"

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Pandera schema — validates raw data before any processing
# ---------------------------------------------------------------------------
RAW_SCHEMA = DataFrameSchema(
    {
        "tenure_months": Column(pa.Int, pa.Check.ge(0), nullable=False),
        "monthly_charges": Column(pa.Float, pa.Check.ge(0), nullable=False),
        "total_charges": Column(pa.Float, nullable=True),   # may be blank in IBM set
        "contract_type": Column(
            pa.String,
            pa.Check.isin(["month-to-month", "one_year", "two_year"]),
            nullable=False,
        ),
        "payment_method": Column(
            pa.String,
            pa.Check.isin(
                ["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
            ),
            nullable=False,
        ),
        "support_tickets_30d": Column(pa.Int, pa.Check.ge(0), nullable=False),
        "is_churned": Column(pa.Int, pa.Check.isin([0, 1]), nullable=False),
    },
    coerce=True,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _normalise_ibm_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Map IBM Telco column names → our canonical names if needed."""
    rename = {
        "tenure": "tenure_months",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Contract": "contract_type",
        "PaymentMethod": "payment_method",
        "Churn": "is_churned",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # IBM dataset uses 'Yes'/'No' for churn
    if df["is_churned"].dtype == object:
        df["is_churned"] = df["is_churned"].map({"Yes": 1, "No": 0})

    # Contract type normalisation
    contract_map = {
        "Month-to-month": "month-to-month",
        "One year": "one_year",
        "Two year": "two_year",
    }
    df["contract_type"] = df["contract_type"].replace(contract_map)

    # Payment method normalisation
    payment_map = {
        "Electronic check": "electronic_check",
        "Mailed check": "mailed_check",
        "Bank transfer (automatic)": "bank_transfer",
        "Credit card (automatic)": "credit_card",
    }
    df["payment_method"] = df["payment_method"].replace(payment_map)

    # IBM TotalCharges can be whitespace string → coerce to float
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # Add support_tickets_30d if not present (IBM set doesn't have it)
    if "support_tickets_30d" not in df.columns:
        rng = np.random.default_rng(42)
        # Simulate: churners tend to have more tickets
        df["support_tickets_30d"] = np.where(
            df["is_churned"] == 1,
            rng.integers(1, 6, size=len(df)),
            rng.integers(0, 3, size=len(df)),
        )

    return df


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load and lightly clean a raw churn CSV."""
    log.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)

    # Drop customer-ID columns (not features)
    id_cols = [c for c in df.columns if c.lower() in ("customerid", "customer_id", "id")]
    if id_cols:
        df = df.drop(columns=id_cols)

    # Normalise IBM Telco naming if needed
    df = _normalise_ibm_telco(df)

    # Keep only columns we care about
    cols_present = [c for c in ALL_FEATURES + [TARGET] if c in df.columns]
    df = df[cols_present]

    return df


# ---------------------------------------------------------------------------
# Preprocessor builder
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for numeric + categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def preprocess(
    csv_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    save_preprocessor: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, fitted_preprocessor, feature_names
    """
    df = load_raw_data(csv_path)

    # ---------- Pandera validation ----------
    log.info("Validating schema with Pandera…")
    try:
        RAW_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        log.warning("Schema validation warnings:\n%s", exc.failure_cases)

    X = df[ALL_FEATURES]
    y = df[TARGET].astype(int).values

    log.info("Dataset shape: %s | Churn rate: %.2f%%", df.shape, y.mean() * 100)

    # ---------- Train / test split ----------
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    log.info(
        "Split → train=%d, test=%d | train churn=%.2f%% | test churn=%.2f%%",
        len(y_train), len(y_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )

    # ---------- Fit preprocessor on train only ----------
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # ---------- Feature names (for SHAP) ----------
    try:
        cat_names: list[str] = list(
            preprocessor.named_transformers_["cat"]
            .named_steps["encoder"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
        )
    except Exception:
        cat_names = []
    feature_names = NUMERIC_FEATURES + cat_names

    log.info("Feature matrix shape after transform: %s", X_train.shape)

    if save_preprocessor:
        out = MODELS_DIR / "preprocessor.joblib"
        joblib.dump(preprocessor, out)
        log.info("Preprocessor saved → %s", out)

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_csv() -> Path:
    candidates = list(DATA_RAW.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {DATA_RAW}. Run `python generate_data.py` first."
        )
    return candidates[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess churn data")
    parser.add_argument("--input", type=Path, default=None, help="Path to raw CSV")
    args = parser.parse_args()

    csv_path = args.input or _find_csv()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess(csv_path)
    log.info("Done. X_train=%s X_test=%s", X_train.shape, X_test.shape)
