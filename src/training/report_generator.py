"""Report generation for model training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_model_report(output_path: str | Path, content: str) -> Path:
    """Write a markdown report to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content, encoding="utf-8")
    return output_file
