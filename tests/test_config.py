"""Tests for GitContext configuration."""

from pathlib import Path

from gitcontext.config import get_env, PROJECT_ROOT


def test_project_root_points_to_repository_root():
    """Verify the dotenv loader uses the repository root."""
    assert PROJECT_ROOT == Path(__file__).resolve().parent.parent


def test_environment_variable_defaults(monkeypatch):
    """Verify missing environment variables use application defaults."""
    monkeypatch.delenv("CHUNK_MAX_TOKENS", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    assert get_env("CHUNK_MAX_TOKENS", "512") == "512"
    assert get_env("LLM_MODEL", "gpt-4o-mini") == "gpt-4o-mini"


def test_environment_variables_override_defaults(monkeypatch):
    """Verify environment variables override supplied defaults."""
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "1024")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")

    assert get_env("CHUNK_MAX_TOKENS", "512") == "1024"
    assert get_env("LLM_MODEL", "gpt-4o-mini") == "gpt-4o"
