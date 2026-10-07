"""Text chunking for LangChain documents.

This module currently provides character-based recursive splitting. AST-aware
chunking is intentionally deferred to a later milestone.
"""

from gitcontext.chunking.splitter import chunk_documents

__all__ = ["chunk_documents"]
