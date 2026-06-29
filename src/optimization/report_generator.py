"""Report generation for optimization results."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_optimization_report(output_path: str | Path, content: str) -> Path:
    """Write the optimization markdown report to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content, encoding="utf-8")
    return output_file
