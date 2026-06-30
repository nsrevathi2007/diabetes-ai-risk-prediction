"""LLM client configuration and provider invocation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .provider import BaseLLMProvider, LLMProviderError, OpenAICompatibleProvider, ProviderConfig


class LLMClient:
    """Thin client that keeps provider-specific logic behind an abstraction."""

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        config: ProviderConfig | None = None,
        logger: Any | None = None,
    ) -> None:
        """Initialize the LLM client."""
        self.config = config or load_provider_config()
        self.provider = provider or OpenAICompatibleProvider(self.config, logger=logger)
        self.logger = logger

    @classmethod
    def from_env(cls, env_path: str | Path = ".env", logger: Any | None = None) -> "LLMClient":
        """Create a client from .env and process environment variables."""
        load_dotenv(env_path)
        return cls(config=load_provider_config(), logger=logger)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text through the configured provider."""
        return self.provider.generate(system_prompt, user_prompt)

    def is_configured(self) -> bool:
        """Return whether an API key is configured."""
        return bool(self.config.api_key)


def load_provider_config() -> ProviderConfig:
    """Load OpenAI-compatible provider configuration from environment variables."""
    return ProviderConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=float(os.getenv("TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("MAX_TOKENS", "800")),
    )


__all__ = ["LLMClient", "LLMProviderError", "ProviderConfig", "load_provider_config"]
