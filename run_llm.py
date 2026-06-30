"""Runner for Phase 8 LLM explanation generation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.llm import LLMExplanationGenerator
from src.llm.report_generator import LLMReportGenerator
from src.utils.logging import configure_logger


def main() -> None:
    """Generate LLM or template explanations for recommendation JSON files."""
    start_time = time.perf_counter()
    logger = configure_logger({"log_dir": "logs", "log_file": "llm.log"})

    recommendation_dir = Path("artifacts/recommendations")
    recommendation_paths = sorted(recommendation_dir.glob("patient_*.json"))
    if not recommendation_paths:
        raise FileNotFoundError(
            "No recommendation JSON files found. Run python run_recommendations.py before run_llm.py."
        )

    generator = LLMExplanationGenerator(logger=logger)
    report_generator = LLMReportGenerator(logger=logger)
    outputs = []

    for path in recommendation_paths:
        recommendation_payload = json.loads(path.read_text(encoding="utf-8"))
        result = generator.generate(recommendation_payload)
        report_generator.save_json(result.payload)
        report_generator.write_markdown(result.payload)
        outputs.append(result.payload)

    report_generator.write_integration_report(outputs)
    logger.info("LLM integration completed in %.2fs", time.perf_counter() - start_time)
    print("LLM INTEGRATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
