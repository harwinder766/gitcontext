from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import logfire

# Invisible characters to sanitize / strip
INVISIBLE_CHARS_PATTERN = re.compile(r"[\u200B\u200C\u200D\uFEFF\u00AD\u200E\u200F]")

# Strict prompt-injection heuristic regex:
# Explicitly avoids matching developer keywords (like 'api_key', 'openai', 'create_llm', 'token')
# while blocking adversarial prompt overrides and jailbreaks.
PROMPT_INJECTION_PATTERN = re.compile(
    r"(ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"
    r"|system\s+prompt"
    r"|you\s+are\s+now\s+in\s+developer\s+mode"
    r"|\bDAN\s+mode\b"
    r"|disregard\s+(?:all\s+)?(?:rules|guidelines)"
    r"|bypass\s+(?:all\s+)?(?:safety\s+)?restrictions"
    r"|override\s+(?:all\s+)?(?:previous\s+)?instructions)",
    re.IGNORECASE,
)

MAX_INPUT_WORDS = 2000
MAX_INPUT_CHARS = 10000


@dataclass
class GuardrailScanners:
    """Holds lightweight input and output scanners (0 MB local RAM footprint)."""

    input_scanners: list[Any]
    output_scanners: list[Any]


def create_guardrail_scanners() -> GuardrailScanners:
    """Initialize lightweight deterministic guardrail scanners.

    Replaces heavy local DeBERTa / transformer models (~600MB-1GB RAM) with zero-RAM,
    sub-millisecond deterministic regex and heuristic scanners.
    """
    with logfire.span("Load lightweight zero-RAM guardrails"):
        # Output scanners are empty list: output scanning is a zero-latency no-op
        return GuardrailScanners(
            input_scanners=[INVISIBLE_CHARS_PATTERN, PROMPT_INJECTION_PATTERN],
            output_scanners=[],
        )


def validate_user_input(
    scanners: GuardrailScanners,
    question: str,
) -> tuple[str, bool, dict[str, float]]:
    """Scan and validate incoming user questions before running LangGraph.

    Enforces token length limits, sanitizes invisible Unicode characters, and detects
    prompt injection attempts with zero local ML memory overhead.
    """
    with logfire.span("🛡️ Input Guardrails", question=question) as span:
        results_score: dict[str, float] = {}

        # 1. Sanitize invisible characters
        sanitized_prompt = INVISIBLE_CHARS_PATTERN.sub("", question).strip()

        # 2. Token / Character length limit check
        word_count = len(sanitized_prompt.split())
        if word_count > MAX_INPUT_WORDS or len(sanitized_prompt) > MAX_INPUT_CHARS:
            results_score["token_limit"] = 1.0
            span.set_attributes({"is_valid": False, "reason": "token_limit_exceeded"})
            logfire.warn(
                "Blocked user prompt exceeding length limit",
                word_count=word_count,
                char_count=len(sanitized_prompt),
            )
            return sanitized_prompt, False, results_score

        # 3. Prompt injection heuristic check
        if PROMPT_INJECTION_PATTERN.search(sanitized_prompt):
            results_score["prompt_injection"] = 1.0
            span.set_attributes({"is_valid": False, "reason": "prompt_injection_detected"})
            logfire.warn("Blocked prompt injection attempt in user query")
            return sanitized_prompt, False, results_score

        # Clean input passed all deterministic checks
        span.set_attributes({"is_valid": True, "results_score": results_score})
        return sanitized_prompt, True, results_score


def sanitize_llm_output(
    scanners: GuardrailScanners,
    prompt: str,
    answer: str,
) -> str:
    """Scan and sanitize the LLM response before returning to user.

    Since output_scanners is an empty list, this is a zero-overhead no-op that
    preserves full response speed and 0 MB RAM.
    """
    if not answer or not scanners.output_scanners:
        return answer

    return answer