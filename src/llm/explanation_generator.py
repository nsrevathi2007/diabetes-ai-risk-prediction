"""Orchestrate LLM and template explanation generation."""

from __future__ import annotations

import time
from typing import Any

from .llm_client import LLMClient
from .prompt_builder import PromptBuilder
from .provider import LLMProviderError
from .response_parser import LLMResponseParser
from .schema import LLMGenerationResult
from .safety import SafetyLayer
from .template_generator import TemplateExplanationGenerator


class LLMExplanationGenerator:
    """Generate communication-layer explanations from recommendation JSON."""

    def __init__(
        self,
        client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
        parser: LLMResponseParser | None = None,
        template_generator: TemplateExplanationGenerator | None = None,
        safety_layer: SafetyLayer | None = None,
        logger: Any | None = None,
    ) -> None:
        """Initialize explanation generation dependencies."""
        self.safety_layer = safety_layer or SafetyLayer()
        self.client = client or LLMClient.from_env(logger=logger)
        self.prompt_builder = prompt_builder or PromptBuilder(self.safety_layer, logger=logger)
        self.parser = parser or LLMResponseParser(self.safety_layer)
        self.template_generator = template_generator or TemplateExplanationGenerator()
        self.logger = logger

    def generate(self, recommendation_payload: dict[str, Any]) -> LLMGenerationResult:
        """Generate one explanation, falling back to template mode on any provider issue."""
        start_time = time.perf_counter()
        model_name = self.client.config.model
        if not self.client.is_configured():
            return self._fallback(recommendation_payload, "OPENAI_API_KEY is not configured", model_name, start_time)

        try:
            system_prompt, user_prompt = self.prompt_builder.build(recommendation_payload)
            raw_response = self.client.generate(system_prompt, user_prompt)
            payload = self.parser.parse(raw_response, recommendation_payload, model_name=model_name)
            self._log(
                "Generated LLM explanation for %s in %.2fs",
                recommendation_payload["patient_id"],
                time.perf_counter() - start_time,
            )
            return LLMGenerationResult(payload=payload, raw_response=raw_response)
        except (LLMProviderError, ValueError, KeyError) as exc:
            return self._fallback(recommendation_payload, str(exc), model_name, start_time)

    def _fallback(
        self,
        recommendation_payload: dict[str, Any],
        reason: str,
        model_name: str,
        start_time: float,
    ) -> LLMGenerationResult:
        """Generate deterministic template output after logging fallback activation."""
        self._log("LLM fallback activated for %s: %s", recommendation_payload.get("patient_id"), reason)
        payload = self.template_generator.generate(recommendation_payload, model_name=model_name)
        self._log(
            "Generated template explanation for %s in %.2fs",
            recommendation_payload["patient_id"],
            time.perf_counter() - start_time,
        )
        return LLMGenerationResult(payload=payload, fallback_reason=reason)

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
