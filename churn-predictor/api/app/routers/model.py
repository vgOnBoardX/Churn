"""Model metadata router."""
from fastapi import APIRouter
from app.schemas import ModelMetadata
from app.services.model_service import ModelService

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/metadata", response_model=ModelMetadata, summary="Get model metrics and metadata")
async def get_metadata() -> ModelMetadata:
    meta = ModelService.get().get_metadata()
    return ModelMetadata(**meta)
