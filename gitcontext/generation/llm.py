import os
import logfire
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def create_llm() -> ChatOpenAI:
    api_key = (os.getenv("XKIRO_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("XKIRO_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip() or None
    model = (os.getenv("XKIRO_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()

    with logfire.span(
        "Create LLM: {model}",
        model=model,
        base_url=base_url or "default",
    ):
        if not api_key:
            raise ValueError(
                "Missing required LLM API key. Set XKIRO_API_KEY or OPENAI_API_KEY."
            )

        try:
            return ChatOpenAI(
                model=model,
                api_key=SecretStr(api_key),
                base_url=base_url,
                temperature=0.1,
                max_tokens=2048,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create LLM '{model}': {e}"
            ) from e
        