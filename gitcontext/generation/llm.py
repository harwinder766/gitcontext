import os
from langchain_openai import ChatOpenAI


def create_llm() -> ChatOpenAI:
    api_key = os.getenv("XKIRO_API_KEY")
    base_url = os.getenv("XKIRO_BASE_URL")
    model = os.getenv("XKIRO_MODEL")

    if not all([api_key, base_url, model]):
        raise ValueError("Missing xKiro API configuration.")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_tokens=2048,
    )