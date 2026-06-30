"""Safety utilities for prompt construction and LLM responses."""

from __future__ import annotations

import copy
import re
from typing import Any


class SafetyLayer:
    """Constrain LLM inputs and outputs to communication-only behavior."""

    INJECTION_PATTERNS = (
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"override\s+(the\s+)?system\s+prompt", re.IGNORECASE),
        re.compile(r"you\s+are\s+now", re.IGNORECASE),
        re.compile(r"developer\s+message", re.IGNORECASE),
        re.compile(r"system\s+prompt", re.IGNORECASE),
    )
    UNSAFE_MEDICAL_PATTERNS = (
        re.compile(r"\bprescribe\b", re.IGNORECASE),
        re.compile(r"\bdiagnose\b", re.IGNORECASE),
        re.compile(r"\bstart\s+(taking|using)\b", re.IGNORECASE),
        re.compile(r"\bstop\s+(taking|using)\b", re.IGNORECASE),
    )

    def sanitize_recommendation_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a deep-copied payload with prompt injection text neutralized."""
        return self._sanitize_value(copy.deepcopy(payload))

    def assert_response_safe(
        self,
        explanation: dict[str, Any],
        source_payload: dict[str, Any],
    ) -> None:
        """Validate that an LLM response did not invent or unsafe-expand content."""
        source_titles = {
            str(action.get("title"))
            for action in source_payload.get("priority_actions", [])
        }
        for action in explanation.get("priority_actions", []):
            title = str(action.get("title"))
            if title not in source_titles:
                raise ValueError(f"LLM response invented an unsupported priority action: {title}")

        for key, value in explanation.items():
            if key == "disclaimer":
                continue
            for text in self._iter_strings(value):
                if self.contains_unsafe_medical_instruction(text):
                    raise ValueError("LLM response contained unsafe medical instruction language")

    def contains_prompt_injection(self, text: str) -> bool:
        """Return whether text looks like an instruction override attempt."""
        return any(pattern.search(text) for pattern in self.INJECTION_PATTERNS)

    def contains_unsafe_medical_instruction(self, text: str) -> bool:
        """Return whether generated text contains prohibited medical instruction language."""
        return any(pattern.search(text) for pattern in self.UNSAFE_MEDICAL_PATTERNS)

    def _sanitize_value(self, value: Any) -> Any:
        """Recursively sanitize strings in JSON-compatible structures."""
        if isinstance(value, dict):
            return {key: self._sanitize_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, str):
            if self.contains_prompt_injection(value):
                return "[removed unsafe instruction]"
            return value
        return value

    def _iter_strings(self, value: Any) -> list[str]:
        """Flatten string values from nested response structures."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            strings: list[str] = []
            for item in value:
                strings.extend(self._iter_strings(item))
            return strings
        if isinstance(value, dict):
            strings = []
            for item in value.values():
                strings.extend(self._iter_strings(item))
            return strings
        return []
