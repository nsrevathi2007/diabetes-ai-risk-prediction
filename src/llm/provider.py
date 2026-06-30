"""Provider abstraction for OpenAI-compatible LLM APIs."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider cannot complete a request."""


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for an OpenAI-compatible chat completion provider."""

    api_key: str | None
    model: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.2
    max_tokens: int = 800
    timeout_seconds: int = 45


class BaseLLMProvider(ABC):
    """Interface for provider-specific LLM calls."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return provider text for the given prompts."""


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI-compatible chat completion provider."""

    def __init__(self, config: ProviderConfig, logger: Any | None = None) -> None:
        """Initialize provider configuration."""
        self.config = config
        self.logger = logger

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call a chat completions endpoint and return message content."""
        if not self.config.api_key:
            raise LLMProviderError("OPENAI_API_KEY is not configured")

        start_time = time.perf_counter()
        endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            self._log("Calling LLM provider model %s", self.config.model)
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMProviderError(f"LLM provider call failed: {exc}") from exc

        try:
            parsed = json.loads(response_body)
            content = parsed["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMProviderError("LLM provider returned an unsupported response shape") from exc

        self._log("LLM provider call completed in %.2fs", time.perf_counter() - start_time)
        return str(content)

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
