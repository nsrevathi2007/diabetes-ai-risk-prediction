"""Version 1 API route definitions for the diabetes prediction service."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["diabetes"])


@router.get("/health")
def health_check() -> dict:
    """Health check endpoint for service readiness."""
    return {"status": "ok"}


@router.post("/predict")
def predict_patient_risk() -> dict:
    """Endpoint to receive patient data and return a risk prediction."""
    raise NotImplementedError("Prediction endpoint is not implemented yet.")
