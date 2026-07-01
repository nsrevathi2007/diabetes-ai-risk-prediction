"""Health check endpoint definitions."""

from datetime import datetime

from fastapi import APIRouter, Depends

from ..dependencies import get_settings
from ..schemas.response_models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["health"])
def health(settings: object = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=True,
        llm_available=False,
        timestamp=datetime.utcnow(),
    )
