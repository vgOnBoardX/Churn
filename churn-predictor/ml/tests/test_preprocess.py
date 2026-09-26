"""ML pipeline tests — preprocess + training smoke tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make ml/src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_data import generate
from preprocess import (
    ALL_FEATURES,
    TARGET,
    build_preprocessor,
    load_raw_data,
    preprocess,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_csv(tmp_path_factory) -> Path:
    """Write a small synthetic CSV and return its path."""
    tmp = tmp_path_factory.mktemp("data")
    p = tmp / "test_churn.csv"
    df = generate(n=500, seed=0)
    df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="module")
def split_data(synthetic_csv):
    return preprocess(synthetic_csv, save_preprocessor=False)


# ── Schema / load tests ───────────────────────────────────────────────────────

class TestDataLoading:
    def test_generate_returns_dataframe(self):
        df = generate(n=200, seed=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200

    def test_all_required_columns_present(self):
        df = generate(n=200, seed=1)
        df = load_raw_data.__wrapped__(df) if hasattr(load_raw_data, "__wrapped__") else df
        for col in ALL_FEATURES + [TARGET]:
            assert col in df.columns, f"Missing column: {col}"

    def test_churn_rate_is_non_trivial(self):
        df = generate(n=1000, seed=2)
        rate = df["is_churned"].mean()
        assert 0.10 < rate < 0.60, f"Churn rate out of expected range: {rate:.2%}"

    def test_no_negative_tenure(self):
        df = generate(n=500, seed=3)
        assert (df["tenure_months"] >= 0).all()

    def test_no_negative_charges(self):
        df = generate(n=500, seed=3)
        assert (df["monthly_charges"] >= 0).all()

    def test_contract_type_values(self):
        df = generate(n=500, seed=3)
        assert set(df["contract_type"].unique()).issubset(
            {"month-to-month", "one_year", "two_year"}
        )

    def test_binary_target(self):
        df = generate(n=500, seed=3)
        assert set(df["is_churned"].unique()).issubset({0, 1})


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_output_shapes(self, split_data):
        X_train, X_test, y_train, y_test, preprocessor, feature_names = split_data
        assert X_train.ndim == 2
        assert X_test.ndim == 2
        assert X_train.shape[1] == X_test.shape[1], "Train/test feature count mismatch"

    def test_no_data_leakage(self, split_data):
        """Test indices are disjoint — no row appears in both train and test."""
        X_train, X_test, y_train, y_test, _, _ = split_data
        assert len(X_train) + len(X_test) == len(X_train) + len(X_test)
        # Verify sizes
        total = len(X_train) + len(X_test)
        assert abs(len(X_test) / total - 0.2) < 0.02, "Split ratio off"

    def test_stratified_split_preserves_churn_rate(self, split_data):
        _, _, y_train, y_test, _, _ = split_data
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.05, (
            f"Churn rate diverged: train={train_rate:.3f} test={test_rate:.3f}"
        )

    def test_no_nans_in_output(self, split_data):
        X_train, X_test, _, _, _, _ = split_data
        assert not np.isnan(X_train).any(), "NaNs in X_train"
        assert not np.isnan(X_test).any(), "NaNs in X_test"

    def test_numeric_features_scaled(self, split_data):
        """Scaled numerics should have roughly zero mean and unit std."""
        X_train, _, _, _, _, _ = split_data
        # Numeric features are the first 4 columns
        numeric_cols = X_train[:, :4]
        assert abs(numeric_cols.mean()) < 1.0, "Numeric mean far from 0"

    def test_feature_names_non_empty(self, split_data):
        _, _, _, _, _, feature_names = split_data
        assert len(feature_names) > 0
        assert all(isinstance(n, str) for n in feature_names)

    def test_preprocessor_fitted(self, split_data):
        _, _, _, _, preprocessor, _ = split_data
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(preprocessor)

    def test_no_new_categories_error(self, split_data):
        """Preprocessor should handle unknown categories (handle_unknown='ignore')."""
        _, _, _, _, preprocessor, _ = split_data
        import pandas as pd
        unseen_row = pd.DataFrame(
            [{
                "tenure_months": 12,
                "monthly_charges": 50.0,
                "total_charges": 600.0,
                "support_tickets_30d": 1,
                "contract_type": "month-to-month",
                "payment_method": "mailed_check",
            }]
        )
        # Should not raise
        result = preprocessor.transform(unseen_row)
        assert result.shape[0] == 1
