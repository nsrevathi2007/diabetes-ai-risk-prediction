"""Parse and validate LLM responses."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .safety import SafetyLayer
from .schema import LLMExplanationPayload


class LLMResponseParser:
    """Parse provider JSON responses into validated explanation payloads."""

    JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(?P<body>.*?)\s*```", re.DOTALL | re.IGNORECASE)

    def __init__(self, safety_layer: SafetyLayer | None = None) -> None:
        """Initialize response parser."""
        self.safety_layer = safety_layer or SafetyLayer()

    def parse(
        self,
        raw_response: str,
        source_payload: dict[str, Any],
        model_name: str,
    ) -> LLMExplanationPayload:
        """Parse, safety-check, and validate an LLM JSON response."""
        parsed = json.loads(self._extract_json_text(raw_response))
        parsed.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        parsed["generation_mode"] = "llm"
        parsed["model"] = model_name
        parsed["source_recommendation_count"] = sum(
            len(entries)
            for entries in source_payload.get("recommendations", {}).values()
        )
        parsed.setdefault("disclaimer", source_payload["disclaimer"])
        self.safety_layer.assert_response_safe(parsed, source_payload)
        return LLMExplanationPayload.model_validate(parsed)

    def _extract_json_text(self, raw_response: str) -> str:
        """Extract JSON content from plain or fenced provider output."""
        stripped = raw_response.strip()
        fence_match = self.JSON_FENCE_PATTERN.search(stripped)
        if fence_match:
            return fence_match.group("body")
        return stripped
