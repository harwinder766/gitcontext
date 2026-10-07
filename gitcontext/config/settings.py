"""Application settings loaded from environment variables.

This module defines all configurable parameters for GitContext.
Settings are loaded from environment variables, with fallback to a .env file.

Key concepts:
- pydantic-settings automatically reads env vars matching the field names
  (case-insensitive, with optional prefix).
- @lru_cache on get_settings() ensures we parse env vars only once.
- All paths are resolved relative to the project root.

Adding a new setting:
1. Add a field to the Settings class with a type hint and default value.
2. Document it in .env.example.
3. Use it via: get_settings().your_new_field
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root is two levels up from this file:
# rag_project/gitcontext/config/settings.py → rag_project/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Central configuration for GitContext.

    All settings can be overridden via environment variables.
    For example, REPO_STORAGE_DIR=./my_repos will override the default.
    """

    model_config = SettingsConfigDict(
        # Load from .env file in the project root
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        # Don't fail if .env doesn't exist
        extra="ignore",
    )

    # --- Repository Storage (Phase 1) ---
    repo_storage_dir: Path = PROJECT_ROOT / "data" / "repos"

    # --- Chunking (Phase 2) ---
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # --- Embeddings (Phase 3) ---
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Vector Store (Phase 3) ---
    chroma_persist_dir: Path = PROJECT_ROOT / "data" / "vectordb"

    # --- Retrieval (Phase 4) ---
    retrieval_top_k: int = 10
    rerank_top_k: int = 5

    # --- Generation (Phase 5) ---
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    anthropic_api_key: str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings.

    Uses @lru_cache so environment variables are parsed exactly once.
    Call get_settings.cache_clear() in tests to reset.
    """
    return Settings()
