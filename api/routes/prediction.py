"""Prediction endpoint definitions."""

from fastapi import APIRouter, Depends

from ..dependencies import get_prediction_service
from ..schemas.request_models import PatientFeatures
from ..schemas.response_models import PredictionResponse

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(features: PatientFeatures, prediction_service = Depends(get_prediction_service)) -> dict[str, object]:
    return prediction_service.predict(features.model_dump())
