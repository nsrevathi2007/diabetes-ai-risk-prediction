"""Full analysis endpoint definitions."""

from fastapi import APIRouter, Depends

from ..dependencies import get_analysis_service
from ..schemas.request_models import PatientFeatures
from ..schemas.response_models import FullAnalysisResponse

router = APIRouter()

@router.post("/full-analysis", response_model=FullAnalysisResponse, tags=["analysis"])
def full_analysis(features: PatientFeatures, analysis_service = Depends(get_analysis_service)) -> dict[str, object]:
    return analysis_service.full_analysis(features.model_dump())
