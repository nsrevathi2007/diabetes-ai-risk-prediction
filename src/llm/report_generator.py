"""Report generation for LLM explanation outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .schema import LLMExplanationPayload


class LLMReportGenerator:
    """Persist LLM explanation JSON, markdown, and docs reports."""

    def __init__(
        self,
        json_dir: str | Path = "artifacts/llm",
        report_dir: str | Path = "reports/llm",
        logger: Any | None = None,
    ) -> None:
        """Initialize report paths."""
        self.json_dir = Path(json_dir)
        self.report_dir = Path(report_dir)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def save_json(self, payload: LLMExplanationPayload) -> Path:
        """Save one explanation payload as JSON."""
        start_time = time.perf_counter()
        path = self.json_dir / f"{payload.patient_id}.json"
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        self._log("Generated LLM JSON at %s in %.2fs", path, time.perf_counter() - start_time)
        return path

    def write_markdown(self, payload: LLMExplanationPayload) -> Path:
        """Write one explanation markdown report."""
        start_time = time.perf_counter()
        path = self.report_dir / f"{payload.patient_id}.md"
        lines = [
            f"# LLM Explanation: {payload.patient_id}",
            "",
            "## Risk Level",
            payload.risk_level,
            "",
            "## Summary",
            payload.summary,
            "",
            "## Why This Prediction",
            payload.why_this_prediction,
            "",
            "## Positive Observations",
            *[f"- {item}" for item in payload.positive_observations],
            "",
            "## Priority Actions",
            *[f"- [{action.priority}] {action.title}: {action.explanation}" for action in payload.priority_actions],
            "",
            "## Recommendation Explanation",
            payload.recommendation_explanation,
            "",
            "## Next Steps",
            payload.next_steps,
            "",
            "## Generation Metadata",
            f"- Mode: {payload.generation_mode}",
            f"- Model: {payload.model}",
            f"- Generated at: {payload.generated_at.isoformat()}",
            "",
            "## Disclaimer",
            payload.disclaimer,
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        self._log("Generated LLM markdown at %s in %.2fs", path, time.perf_counter() - start_time)
        return path

    def write_integration_report(
        self,
        payloads: list[LLMExplanationPayload],
        output_path: str | Path = "docs/llm_integration_report.md",
    ) -> Path:
        """Write the Phase 8 LLM integration documentation report."""
        path = Path(output_path)
        template_count = sum(1 for payload in payloads if payload.generation_mode == "template")
        llm_count = sum(1 for payload in payloads if payload.generation_mode == "llm")
        example = payloads[0].model_dump(mode="json") if payloads else {}
        lines = [
            "# LLM Integration Report",
            "",
            "## Architecture",
            "Recommendation Engine -> Structured Recommendation JSON -> Prompt Builder -> LLM Provider or Template Mode -> Structured Explanation JSON.",
            "",
            "The recommendation engine remains the source of truth. The LLM layer only explains existing predictions, SHAP factors, observations, and recommendations.",
            "",
            "## Prompt Design",
            "Prompts are split into system and user prompts. The user prompt includes only whitelisted recommendation fields and avoids duplicate SHAP detail.",
            "",
            "## Offline Mode",
            f"Template Mode generated {template_count} explanation outputs. LLM mode generated {llm_count} explanation outputs.",
            "",
            "## Safety",
            "The system prompt prohibits diagnosis, prescriptions, replacing healthcare professionals, and invented recommendations. The safety layer removes prompt-injection text and rejects unsupported generated priority actions.",
            "",
            "## Example Outputs",
            "```json",
            json.dumps(example, indent=2),
            "```",
            "",
            "## Future Improvements",
            "- Add provider adapters for Azure OpenAI, Ollama, Gemini, and other OpenAI-compatible services.",
            "- Persist provider latency and fallback metrics for monitoring.",
            "- Add frontend controls for explanation tone and detail level while preserving the same structured source JSON.",
            "",
            "## Disclaimer",
            (
                "This information is generated for educational purposes only and should not be considered medical advice. "
                "Please consult a qualified healthcare professional for diagnosis or treatment."
            ),
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        self._log("Generated LLM integration report at %s", path)
        return path

    def _log(self, message: str, *args: Any) -> None:
        """Log when a project logger is available."""
        if self.logger is not None:
            self.logger.info(message, *args)
