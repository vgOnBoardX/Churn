"""Health check router."""
from fastapi import APIRouter
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])

API_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=API_VERSION)
