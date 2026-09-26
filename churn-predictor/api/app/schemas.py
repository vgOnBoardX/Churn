"""
Pydantic v2 schemas for the Churn Prediction API.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


# ── Input ─────────────────────────────────────────────────────────────────────

class CustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str | None = Field(
        default=None, description="Optional customer identifier"
    )
    tenure_months: Annotated[int, Field(ge=0, le=720, description="Months as customer")]
    monthly_charges: Annotated[
        float, Field(ge=0.0, le=500.0, description="Monthly bill in USD")
    ]
    total_charges: Annotated[
        float, Field(ge=0.0, description="Cumulative charges in USD")
    ]
    contract_type: Literal["month-to-month", "one_year", "two_year"] = Field(
        description="Contract term"
    )
    payment_method: Literal[
        "electronic_check", "mailed_check", "bank_transfer", "credit_card"
    ] = Field(description="Payment method")
    support_tickets_30d: Annotated[
        int, Field(ge=0, le=100, description="Support tickets in last 30 days")
    ]


# ── Output ────────────────────────────────────────────────────────────────────

class ShapFactor(BaseModel):
    feature: str
    value: float | None
    shap_impact: float
    direction: Literal["increases_churn", "decreases_churn"]


class PredictionResponse(BaseModel):
    customer_id: str | None = None
    churn_probability: float = Field(ge=0.0, le=1.0)
    risk_tier: Literal["low", "medium", "high"]
    top_factors: list[ShapFactor]


class BatchPredictionRow(BaseModel):
    row_index: int
    customer_id: str | None = None
    churn_probability: float
    risk_tier: Literal["low", "medium", "high"]
    top_factors: list[ShapFactor]


class BatchPredictionResponse(BaseModel):
    total_rows: int
    predictions: list[BatchPredictionRow]


# ── Model metadata ────────────────────────────────────────────────────────────

class ModelMetadata(BaseModel):
    model_version: str
    trained_at: str
    roc_auc: float
    avg_precision: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    test_samples: int
    churn_rate_test: float


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


# ── At-risk customers ─────────────────────────────────────────────────────────

class AtRiskCustomer(BaseModel):
    id: str
    name: str
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    payment_method: str
    support_tickets_30d: int
    churn_probability: float
    risk_tier: Literal["low", "medium", "high"]
    top_factors: list[ShapFactor]


class AtRiskResponse(BaseModel):
    total: int
    customers: list[AtRiskCustomer]
