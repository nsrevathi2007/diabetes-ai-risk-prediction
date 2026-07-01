"""Explanation endpoint definitions."""

from fastapi import APIRouter, Depends

from ..dependencies import get_explanation_service
from ..schemas.request_models import PatientFeatures
from ..schemas.response_models import ExplanationResponse

router = APIRouter()

@router.post("/explain", response_model=ExplanationResponse, tags=["explanation"])
def explain(features: PatientFeatures, explanation_service = Depends(get_explanation_service)) -> dict[str, object]:
    return explanation_service.explain(features.model_dump())
