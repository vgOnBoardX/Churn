"""
generate_data.py
================
Generates a synthetic customer churn dataset with realistic correlations.
Run this if no raw CSV exists in ml/data/raw/.

Usage:
    python generate_data.py [--rows 5000] [--seed 42]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "raw" / "synthetic_churn.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Customer attributes ----
    tenure_months = rng.integers(1, 73, size=n)          # 1–72 months
    monthly_charges = rng.uniform(20.0, 120.0, size=n)
    total_charges = tenure_months * monthly_charges * rng.uniform(0.95, 1.05, size=n)

    contract_type = rng.choice(
        ["month-to-month", "one_year", "two_year"],
        p=[0.55, 0.25, 0.20],
        size=n,
    )
    payment_method = rng.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
        p=[0.34, 0.23, 0.22, 0.21],
        size=n,
    )
    support_tickets_30d = rng.integers(0, 8, size=n)

    # ---- Churn probability (engineered signal) ----
    logit = np.zeros(n)

    # Month-to-month is big churn driver
    logit += np.where(contract_type == "month-to-month", 1.5, 0.0)
    logit += np.where(contract_type == "one_year", 0.1, 0.0)

    # Short tenure → higher churn
    logit += np.where(tenure_months < 6, 1.2, 0.0)
    logit += np.where((tenure_months >= 6) & (tenure_months < 12), 0.6, 0.0)
    logit += np.where((tenure_months >= 12) & (tenure_months < 24), 0.1, 0.0)
    logit += np.where(tenure_months >= 48, -0.8, 0.0)

    # High support tickets → churn
    logit += np.where(support_tickets_30d >= 4, 1.0, 0.0)
    logit += np.where(support_tickets_30d == 3, 0.5, 0.0)

    # High monthly charges (price sensitivity)
    logit += np.where(monthly_charges > 90, 0.4, 0.0)
    logit += np.where(monthly_charges < 35, -0.3, 0.0)

    # Electronic check payment → slight churn correlation
    logit += np.where(payment_method == "electronic_check", 0.3, 0.0)

    # Intercept to get ~28% base churn rate
    logit += -1.8

    # Add noise
    logit += rng.normal(0, 0.5, size=n)

    # Sigmoid → probability → binary label
    prob = 1 / (1 + np.exp(-logit))
    is_churned = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(n)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges.round(2),
            "total_charges": total_charges.round(2),
            "contract_type": contract_type,
            "payment_method": payment_method,
            "support_tickets_30d": support_tickets_30d,
            "is_churned": is_churned,
        }
    )

    churn_rate = is_churned.mean() * 100
    log.info(
        "Generated %d rows | churn rate: %.1f%% | shape: %s",
        n, churn_rate, df.shape,
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    df = generate(n=args.rows, seed=args.seed)
    df.to_csv(args.output, index=False)
    log.info("Saved → %s", args.output)
