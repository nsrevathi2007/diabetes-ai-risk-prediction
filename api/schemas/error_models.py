"""Structured API error models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard API error shape."""

    success: bool = False
    error: str
    code: int
    details: list[dict[str, Any]] | None = None
