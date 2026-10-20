from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_guard import scan_output, scan_prompt
from llm_guard.input_scanners import InvisibleText, PromptInjection, TokenLimit
from llm_guard.output_scanners import Toxicity
import logfire


@dataclass
class GuardrailScanners:
    """Holds initialized input and output scanners."""

    input_scanners: list[Any]
    output_scanners: list[Any]


def create_guardrail_scanners() -> GuardrailScanners:
    """Initialize LLM Guard scanners once at application startup."""
    with logfire.span("Load LLM Guard scanners"):
        input_scanners = [
            InvisibleText(),
            TokenLimit(limit=2000),
            PromptInjection(threshold=0.80, use_onnx=True),
        ]
        
        # Keep only Toxicity or an empty list [] for output scanners
        output_scanners = [
            Toxicity(threshold=0.80, use_onnx=True),
        ]
        
        return GuardrailScanners(
            input_scanners=input_scanners,
            output_scanners=output_scanners,
        )


def validate_user_input(
    scanners: GuardrailScanners,
    question: str,
) -> tuple[str, bool, dict[str, float]]:
    """Scan the incoming user question before running LangGraph."""
    with logfire.span("🛡️ Input Guardrails", question=question) as span:
        sanitized_prompt, results_valid, results_score = scan_prompt(
            scanners.input_scanners,
            question,
        )
        is_valid = all(results_valid.values())
        span.set_attributes(
            {
                "is_valid": is_valid,
                "results_valid": results_valid,
                "risk_scores": results_score,
            }
        )
        if not is_valid:
            logfire.warn(
                "Blocked unsafe user prompt",
                results_valid=results_valid,
                risk_scores=results_score,
            )
        return sanitized_prompt, is_valid, results_score


def sanitize_llm_output(
    scanners: GuardrailScanners,
    prompt: str,
    answer: str,
) -> str:
    """Scan and redact the LLM's response before returning it to the user."""
    if not answer or not scanners.output_scanners:
        return answer

    with logfire.span("🛡️ Output Guardrails") as span:
        sanitized_answer, results_valid, results_score = scan_output(
            scanners.output_scanners,
            prompt,
            answer,
        )
        is_valid = all(results_valid.values())
        span.set_attributes(
            {
                "is_valid": is_valid,
                "results_valid": results_valid,
                "risk_scores": results_score,
            }
        )
        if not is_valid:
            logfire.warn(
                "Output modified or flagged by guardrails",
                results_valid=results_valid,
                risk_scores=results_score,
            )
        return sanitized_answer