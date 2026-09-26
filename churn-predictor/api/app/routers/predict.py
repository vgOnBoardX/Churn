"""
predict.py
==========
POST /predict        — single customer prediction
POST /predict/batch  — CSV batch upload
"""
from __future__ import annotations

import csv
import io
import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from app.schemas import (
    BatchPredictionResponse,
    BatchPredictionRow,
    CustomerInput,
    PredictionResponse,
    ShapFactor,
)
from app.services.model_service import ModelService

log = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predictions"])


def _build_prediction_response(raw: dict) -> PredictionResponse:
    return PredictionResponse(
        customer_id=raw.get("customer_id"),
        churn_probability=raw["churn_probability"],
        risk_tier=raw["risk_tier"],
        top_factors=[ShapFactor(**f) for f in raw["top_factors"]],
    )


@router.post("", response_model=PredictionResponse, summary="Single customer churn prediction")
async def predict_single(customer: CustomerInput) -> PredictionResponse:
    """
    Given a single customer record, return:
    - `churn_probability` (0–1)
    - `risk_tier` (low / medium / high)
    - `top_factors` — top-3 SHAP-driven explanations
    """
    try:
        svc = ModelService.get()
        raw = svc.predict_single(customer.model_dump())
        return _build_prediction_response(raw)
    except Exception as exc:
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/batch", response_model=BatchPredictionResponse, summary="Batch CSV prediction")
async def predict_batch(file: Annotated[UploadFile, File(description="CSV file with customer records")]) -> BatchPredictionResponse:
    """
    Upload a CSV file with customer records and receive predictions for every row.
    The response JSON also includes a `download_url` hint; the CSV download
    is handled by calling this endpoint with `Accept: text/csv`.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    required = {"tenure_months", "monthly_charges", "total_charges",
                "contract_type", "payment_method", "support_tickets_30d"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV missing required columns: {sorted(missing)}",
        )

    try:
        svc = ModelService.get()
        rows = svc.predict_batch(df)
    except Exception as exc:
        log.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    predictions = [
        BatchPredictionRow(
            row_index=r["row_index"],
            customer_id=r.get("customer_id"),
            churn_probability=r["churn_probability"],
            risk_tier=r["risk_tier"],
            top_factors=[ShapFactor(**f) for f in r["top_factors"]],
        )
        for r in rows
    ]

    return BatchPredictionResponse(total_rows=len(predictions), predictions=predictions)


@router.post("/batch/download", summary="Batch CSV prediction — returns downloadable CSV")
async def predict_batch_csv(
    file: Annotated[UploadFile, File(description="CSV file with customer records")]
) -> StreamingResponse:
    """Returns predictions as a downloadable CSV file."""
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    svc = ModelService.get()
    rows = svc.predict_batch(df)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["row_index", "customer_id", "churn_probability", "risk_tier", "top_feature_1", "impact_1"])
    for r in rows:
        top = r["top_factors"][0] if r["top_factors"] else {}
        writer.writerow([
            r["row_index"],
            r.get("customer_id", ""),
            r["churn_probability"],
            r["risk_tier"],
            top.get("feature", ""),
            top.get("shap_impact", ""),
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"},
    )
