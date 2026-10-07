"""Repository ingestion — file discovery and loading.

This module handles the first stage of the RAG pipeline: finding and reading
relevant files from a repository directory.

Usage:
    from gitcontext.ingestion import discover_files

    docs = discover_files(Path("/path/to/repo"))
"""

from gitcontext.ingestion.discovery import discover_files

__all__ = ["discover_files"]
