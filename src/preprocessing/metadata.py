"""Metadata helpers for preprocessing artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import pandas as pd


class PreprocessingMetadata:
    """Persist and retrieve metadata for preprocessing artifacts."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        """Initialize the metadata manager."""
        self.output_dir = Path(output_dir or "artifacts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, metadata: dict[str, Any], filename: str) -> Path:
        """Save metadata as JSON to disk."""
        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return output_path

    def load_metadata(self, filename: str) -> dict[str, Any]:
        """Load metadata from JSON on disk."""
        return json.loads((self.output_dir / filename).read_text(encoding="utf-8"))
