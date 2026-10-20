import os
import logfire
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def create_llm() -> ChatOpenAI:
    api_key = (os.getenv("XKIRO_API_KEY") or "").strip()
    base_url = (os.getenv("XKIRO_BASE_URL") or "").strip()
    model = (os.getenv("XKIRO_MODEL") or "").strip()

    with logfire.span(
        "Create xKiro LLM: {model}",
        model=model,
        base_url=base_url,
    ):
        missing = [
            name
            for name, val in [
                ("XKIRO_API_KEY", api_key),
                ("XKIRO_BASE_URL", base_url),
                ("XKIRO_MODEL", model),
            ]
            if not val
        ]
        if missing:
            raise ValueError(
                f"Missing required xKiro environment variables: {', '.join(missing)}"
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
        