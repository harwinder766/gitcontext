"""Tests for GitContext configuration."""

from gitcontext.config.settings import Settings, get_settings


def test_default_settings_load():
    """Verify that Settings can be instantiated with defaults."""
    # Clear the cache so we get a fresh instance
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.chunk_max_tokens == 512
    assert settings.chunk_overlap_tokens == 50
    assert settings.retrieval_top_k == 10
    assert settings.embedding_provider == "huggingface"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.llm_model == "gpt-4o-mini"


def test_settings_repo_storage_dir_is_path():
    """Verify repo_storage_dir is a Path object, not a string."""
    get_settings.cache_clear()
    settings = get_settings()

    from pathlib import Path
    assert isinstance(settings.repo_storage_dir, Path)
    assert str(settings.repo_storage_dir).endswith("repos")


def test_settings_override_via_env(monkeypatch):
    """Verify that environment variables override defaults."""
    get_settings.cache_clear()
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "1024")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")

    settings = Settings()  # Create fresh, not cached
    assert settings.chunk_max_tokens == 1024
    assert settings.llm_model == "gpt-4o"
