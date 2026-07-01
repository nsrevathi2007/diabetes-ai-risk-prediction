"""Model metadata endpoint definitions."""

from fastapi import APIRouter

from ..schemas.response_models import ModelInfoResponse

router = APIRouter()

@router.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name="Best Available Model",
        training_date="unknown",
        metrics={},
        feature_count=22,
        model_version="1.0",
    )
