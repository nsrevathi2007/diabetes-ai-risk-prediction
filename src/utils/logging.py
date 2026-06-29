"""Logging configuration helpers for application observability."""

from __future__ import annotations

import logging
from logging import Logger
from pathlib import Path
from typing import Any, Dict


def configure_logger(config: Dict[str, Any] | None = None) -> Logger:
    """Set up structured logging and return a logger instance.

    Args:
        config: Optional dictionary with ``log_dir``, ``log_file``, ``level``, and
            ``format`` entries.

    Returns:
        A configured logger instance.
    """
    config = config or {}
    log_dir = Path(config.get("log_dir", "logs"))
    log_file = config.get("log_file", "preprocessing.log")
    log_level = getattr(logging, str(config.get("level", "INFO")).upper(), logging.INFO)
    log_format = config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("diabetes_ai")
    logger.setLevel(log_level)
    logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
