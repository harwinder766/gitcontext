import pytest
import litellm
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_litellm import ChatLiteLLMRouter
from pydantic import BaseModel

from gitcontext.generation.llm import create_llm


def test_create_llm_raises_when_no_api_keys(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("XKIRO_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Neither GROQ_API_KEY nor OPENAI_API_KEY is configured"):
        create_llm()


def test_create_llm_with_openai_only(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key")
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-4o-mini")

    llm = create_llm()

    assert isinstance(llm, BaseChatModel)
    assert isinstance(llm, ChatLiteLLMRouter)
    assert llm.model == "gitcontext-primary"

    # Check router deployment list
    router = llm.router
    deployments = router.model_list
    assert len(deployments) == 1
    assert deployments[0]["model_name"] == "gitcontext-primary"
    assert deployments[0]["litellm_params"]["model"] == "openai/gpt-4o-mini"
    assert deployments[0]["litellm_params"]["api_key"] == "sk-openai-test-key"
    assert not router.fallbacks
    assert router.cache_responses is True


def test_create_llm_with_groq_only(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-groq-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("XKIRO_API_KEY", raising=False)

    llm = create_llm()

    assert isinstance(llm, ChatLiteLLMRouter)
    router = llm.router
    deployments = router.model_list
    assert len(deployments) == 1
    assert deployments[0]["model_name"] == "gitcontext-primary"
    assert deployments[0]["litellm_params"]["model"] == "groq/llama-3.3-70b-versatile"
    assert deployments[0]["litellm_params"]["api_key"] == "gsk-groq-test-key"
    assert not router.fallbacks


def test_create_llm_with_both_keys_sets_fallback(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-groq-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key")

    llm = create_llm(temperature=0.2)

    assert isinstance(llm, ChatLiteLLMRouter)
    assert llm.temperature == 0.2

    router = llm.router
    deployments = router.model_list
    assert len(deployments) == 2

    # Primary should be Groq
    assert deployments[0]["model_name"] == "gitcontext-primary"
    assert deployments[0]["litellm_params"]["model"] == "groq/llama-3.3-70b-versatile"
    assert deployments[0]["litellm_params"]["api_key"] == "gsk-groq-test-key"

    # Fallback should be OpenAI
    assert deployments[1]["model_name"] == "gitcontext-fallback"
    assert deployments[1]["litellm_params"]["model"] == "openai/gpt-4o-mini"
    assert deployments[1]["litellm_params"]["api_key"] == "sk-openai-test-key"

    # Fallbacks configuration
    assert router.fallbacks == [{"gitcontext-primary": ["gitcontext-fallback"]}]
    assert router.num_retries == 2
    assert router.timeout == 30


def test_create_llm_configures_caching(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LITELLM_CACHE_TYPE", "local")
    monkeypatch.setenv("LITELLM_CACHE_TTL", "7200")

    create_llm()

    assert litellm.cache is not None
    assert litellm.cache.type == "local"


def test_chat_litellm_router_structured_output_binding(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = create_llm()

    class SampleSchema(BaseModel):
        summary: str

    structured = llm.with_structured_output(SampleSchema)
    assert structured is not None
