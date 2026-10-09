"""Environment-based configuration for GitContext."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_env(name: str, default: str) -> str:
    """Return an environment variable, falling back to the supplied default."""
    return os.getenv(name, default)
