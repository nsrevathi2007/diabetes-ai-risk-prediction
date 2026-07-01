"""FastAPI middleware setup."""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware


def register_middleware(app: FastAPI, logger: Any | None = None) -> None:
    """Register request logging, timing, CORS, and trusted host middleware."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Any:
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        response.headers["X-Execution-Time"] = f"{duration:.4f}"
        if logger is not None:
            logger.info(
                "%s %s completed with %s in %.4fs",
                request.method,
                request.url.path,
                response.status_code,
                duration,
            )
        return response
