"""API route module aggregators."""

from fastapi import APIRouter

from .analysis import router as analysis_router
from .explanation import router as explanation_router
from .health import router as health_router
from .model import router as model_router
from .prediction import router as prediction_router
from .recommendation import router as recommendation_router
from .root import router as root_router

api_router = APIRouter(prefix="/api/v1", tags=["diabetes"])
api_router.include_router(health_router)
api_router.include_router(model_router)
api_router.include_router(prediction_router)
api_router.include_router(recommendation_router)
api_router.include_router(explanation_router)
api_router.include_router(analysis_router)

__all__ = ["api_router", "root_router"]
