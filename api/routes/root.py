"""Root route definitions for API metadata."""

from fastapi import APIRouter

router = APIRouter()

@router.get("/", tags=["root"])
def root() -> dict[str, str]:
    return {"message": "Diabetes Risk Prediction API is running."}
