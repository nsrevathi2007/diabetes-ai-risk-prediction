"""LLM communication layer for recommendation explanations."""

from .explanation_generator import LLMExplanationGenerator
from .prompt_builder import PromptBuilder
from .schema import LLMExplanationPayload
from .template_generator import TemplateExplanationGenerator

__all__ = [
    "LLMExplanationGenerator",
    "LLMExplanationPayload",
    "PromptBuilder",
    "TemplateExplanationGenerator",
]
