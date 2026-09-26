"""
customers.py
============
GET /customers/at-risk?limit=50  — top-N highest-risk customers
GET /customers/{id}              — single customer detail
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.schemas import AtRiskCustomer, AtRiskResponse, ShapFactor
from app.services.db_service import get_at_risk_customers, get_customer_by_id

router = APIRouter(prefix="/customers", tags=["customers"])


def _to_schema(c) -> AtRiskCustomer:
    factors = c.top_factors or []
    return AtRiskCustomer(
        id=c.id,
        name=c.name,
        tenure_months=c.tenure_months,
        monthly_charges=c.monthly_charges,
        total_charges=c.total_charges,
        contract_type=c.contract_type,
        payment_method=c.payment_method,
        support_tickets_30d=c.support_tickets_30d,
        churn_probability=c.churn_probability,
        risk_tier=c.risk_tier,
        top_factors=[ShapFactor(**f) for f in factors],
    )


@router.get("/at-risk", response_model=AtRiskResponse, summary="Top at-risk customers")
async def at_risk_customers(
    limit: Annotated[int, Query(ge=1, le=200, description="Max customers to return")] = 50,
) -> AtRiskResponse:
    """Returns the top N customers sorted by churn probability descending."""
    customers = await get_at_risk_customers(limit=limit)
    return AtRiskResponse(
        total=len(customers),
        customers=[_to_schema(c) for c in customers],
    )


@router.get("/{customer_id}", response_model=AtRiskCustomer, summary="Customer detail")
async def get_customer(customer_id: str) -> AtRiskCustomer:
    c = await get_customer_by_id(customer_id)
    if not c:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id!r} not found.")
    return _to_schema(c)
