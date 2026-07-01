"""Tests for the Streamlit frontend integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "frontend" / "web" / "app.py"


def test_streamlit_app_exists() -> None:
    assert APP_PATH.exists(), "Streamlit app module should exist"


def test_streamlit_app_imports() -> None:
    spec = importlib.util.spec_from_file_location("streamlit_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "FEATURE_FIELDS")
    assert hasattr(module, "render_sidebar")
