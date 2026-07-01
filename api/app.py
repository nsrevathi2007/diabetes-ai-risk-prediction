"""FastAPI application factory for the diabetes prediction backend."""

from __future__ import annotations

from fastapi import FastAPI

from .config import load_settings
from .exceptions import register_exception_handlers
from .middleware import register_middleware
from .routes import api_router, root_router
from .startup import initialize_app_state


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = load_settings()
    app = FastAPI(
        title="Diabetes Risk Prediction API",
        version=settings.api_version,
        description=(
            "AI-powered diabetes risk prediction and personalized health recommendation "
            "service with explainability and optional LLM-generated guidance."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    register_middleware(app)
    register_exception_handlers(app)

    app.include_router(root_router)
    app.include_router(api_router)

    initialize_app_state(app, settings)
    return app
