"""Recommendation endpoint definitions."""

from fastapi import APIRouter, Depends

from ..dependencies import get_recommendation_service
from ..schemas.request_models import PatientFeatures
from ..schemas.response_models import RecommendationResponse

router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse, tags=["recommendation"])
def recommend(features: PatientFeatures, recommendation_service = Depends(get_recommendation_service)) -> dict[str, object]:
    return recommendation_service.recommend(features.model_dump())
