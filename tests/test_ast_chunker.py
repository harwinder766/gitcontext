"""Unit tests for AST Python code chunking."""

from langchain_core.documents import Document
from gitcontext.chunking.ast_chunker import chunk_python_document


def test_chunk_python_function_and_class():
    code = '''"""Module docstring."""
import os

CONSTANT = 42

# Helper function comment
def helper(x: int) -> int:
    """Helper docstring."""
    return x + 1

class Calculator:
    """Calculator class."""

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
'''
    doc = Document(
        page_content=code,
        metadata={"source": "calc.py", "repo_name": "test_repo"},
    )

    chunks = chunk_python_document(doc, max_chars=1000)
    assert chunks is not None
    assert len(chunks) >= 3

    # Check that metadata has symbol information
    symbol_types = {chunk.metadata.get("symbol_type") for chunk in chunks}
    assert "function" in symbol_types or "method" in symbol_types
    assert any(chunk.metadata.get("symbol_name") == "helper" for chunk in chunks)

    # Check header exists in chunks
    assert any("File: calc.py" in chunk.page_content for chunk in chunks)
    assert any("Docstring: Helper docstring." in chunk.page_content for chunk in chunks)


def test_chunk_python_invalid_syntax_returns_none():
    invalid_code = "def invalid_syntax(:"
    doc = Document(page_content=invalid_code, metadata={"source": "bad.py"})

    result = chunk_python_document(doc)
    assert result is None


def test_chunk_python_async_function():
    code = '''async def fetch_data(url: str):
    """Fetch data asynchronously."""
    return url
'''
    doc = Document(page_content=code, metadata={"source": "async.py"})
    chunks = chunk_python_document(doc)

    assert chunks is not None
    assert len(chunks) == 1
    assert chunks[0].metadata["symbol_name"] == "fetch_data"
    assert chunks[0].metadata["symbol_type"] == "function"
