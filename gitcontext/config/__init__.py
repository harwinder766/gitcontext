"""Configuration management for GitContext.

Uses pydantic-settings to load configuration from environment variables
and .env files. All settings have sensible defaults so the project works
out of the box for development.

Usage:
    from gitcontext.config.settings import get_settings

    settings = get_settings()
    print(settings.repo_storage_dir)
"""
