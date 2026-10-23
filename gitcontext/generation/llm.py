from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_litellm import ChatLiteLLMRouter
import litellm
from litellm import Router
from litellm.caching.caching import Cache
import logfire

load_dotenv()


@logfire.instrument("Initialize LiteLLM Router and Cache")
def create_llm(temperature: float | None = None) -> BaseChatModel:
    """Initialize LiteLLM Router with primary/fallback models and caching, wrapped in ChatLiteLLMRouter."""
    if temperature is None:
        try:
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        except (ValueError, TypeError):
            temperature = 0.1

    cache_type = os.getenv("LITELLM_CACHE_TYPE", "local").strip()
    try:
        cache_ttl = int(os.getenv("LITELLM_CACHE_TTL", "3600"))
    except (ValueError, TypeError):
        cache_ttl = 3600

    # Initialize LiteLLM cache
    litellm.cache = Cache(type=cache_type, ttl=cache_ttl)

    groq_api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    gemini_api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    xkiro_api_key = (os.getenv("XKIRO_API_KEY") or "").strip()

    configured_providers: list[dict[str, Any]] = []

    # 1. Groq Provider (Primary fast free model)
    if groq_api_key:
        groq_model = (os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()
        # Clean up common naming mistakes
        groq_model = groq_model.replace("groq/", "")
        configured_providers.append(
            {
                "provider": "groq",
                "litellm_params": {
                    "model": f"groq/{groq_model}",
                    "api_key": groq_api_key,
                },
            }
        )

    # 2. Gemini Provider (Fallback free model)
    if gemini_api_key:
        gemini_model = (os.getenv("GEMINI_MODEL") or "gemini-1.5-flash-latest").strip()
        # Strip prefixes so LiteLLM resolves the standard Google endpoint properly
        gemini_model = gemini_model.replace("gemini/", "").replace("models/", "")
        configured_providers.append(
            {
                "provider": "gemini",
                "litellm_params": {
                    "model": f"gemini/{gemini_model}",
                    "api_key": gemini_api_key,
                },
            }
        )

    # 3. Custom / Proxy Provider (XKIRO / OpenRouter)
    if xkiro_api_key:
        xkiro_model = (os.getenv("XKIRO_MODEL") or "").strip()
        xkiro_base_url = (os.getenv("XKIRO_BASE_URL") or "").strip()

        if xkiro_model and xkiro_base_url:
            if not xkiro_model.startswith("openai/"):
                xkiro_model = f"openai/{xkiro_model}"
            configured_providers.append(
                {
                    "provider": "xkiro",
                    "litellm_params": {
                        "model": xkiro_model,
                        "api_base": xkiro_base_url,
                        "api_key": xkiro_api_key,
                    },
                }
            )

    if not configured_providers:
        raise RuntimeError(
            "None of GROQ_API_KEY, GEMINI_API_KEY, or XKIRO_API_KEY are configured. "
            "At least one valid API key is required to initialize the LiteLLM Router."
        )

    model_list: list[dict[str, Any]] = []
    fallback_names: list[str] = []

    for idx, item in enumerate(configured_providers):
        role_name = (
            "gitcontext-primary"
            if idx == 0
            else f"gitcontext-fallback-{item['provider']}"
        )
        model_list.append(
            {
                "model_name": role_name,
                "litellm_params": item["litellm_params"],
            }
        )
        if idx > 0:
            fallback_names.append(role_name)

    fallbacks: list[dict[str, list[str]]] = (
        [{"gitcontext-primary": fallback_names}] if fallback_names else []
    )

    primary_model = configured_providers[0]["litellm_params"]["model"]
    fallback_models = [p["litellm_params"]["model"] for p in configured_providers[1:]]

    litellm_router = Router(
        model_list=model_list,
        fallbacks=fallbacks,
        timeout=12,
        num_retries=1,
        cache_responses=True,
        cooldown_time=60,
        allowed_fails=1,
    )

    logfire.info(
        "LiteLLM Router initialized: primary={primary_model}, fallbacks={fallback_models}, cache_type={cache_type}, cache_ttl={cache_ttl}",
        primary_model=primary_model,
        fallback_models=fallback_models,
        cache_type=cache_type,
        cache_ttl=cache_ttl,
    )

    return ChatLiteLLMRouter(
        router=litellm_router,
        model_name="gitcontext-primary",
        temperature=temperature,
        caching=True,
    )