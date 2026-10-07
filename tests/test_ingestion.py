"""Tests for repository file discovery.

Tests use pytest's tmp_path fixture to create isolated temporary directories
with controlled file structures. Each test builds a fake repo, calls
discover_files(), and asserts on the returned LangChain Document objects.
"""

import pytest
from langchain_core.documents import Document

from gitcontext.ingestion.discovery import discover_files


# ---------------------------------------------------------------------------
# Happy path: files that SHOULD be discovered
# ---------------------------------------------------------------------------


def test_discovers_python_file(tmp_path):
    """A .py file at the repo root should be discovered."""
    (tmp_path / "main.py").write_text("def hello(): pass", encoding="utf-8")

    docs = discover_files(tmp_path)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "def hello(): pass"
    assert docs[0].metadata["source"] == "main.py"
    assert docs[0].metadata["extension"] == ".py"


def test_discovers_markdown_file(tmp_path):
    """A .md file should be discovered."""
    (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")

    docs = discover_files(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "README.md"
    assert docs[0].metadata["extension"] == ".md"


def test_discovers_all_supported_extensions(tmp_path):
    """All four supported file types (.py, .md, .rst, .txt) should be found."""
    (tmp_path / "code.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# Readme", encoding="utf-8")
    (tmp_path / "notes.rst").write_text("Notes", encoding="utf-8")
    (tmp_path / "todo.txt").write_text("Todo", encoding="utf-8")

    docs = discover_files(tmp_path)

    extensions = {doc.metadata["extension"] for doc in docs}
    assert extensions == {".py", ".md", ".rst", ".txt"}
    assert len(docs) == 4


def test_discovers_nested_files(tmp_path):
    """Files in subdirectories should have correct relative paths using forward slashes."""
    nested_dir = tmp_path / "src" / "utils"
    nested_dir.mkdir(parents=True)
    (nested_dir / "helpers.py").write_text("def help(): pass", encoding="utf-8")

    docs = discover_files(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "src/utils/helpers.py"


# ---------------------------------------------------------------------------
# Exclusion: files that should NOT be discovered
# ---------------------------------------------------------------------------


def test_skips_excluded_directories(tmp_path):
    """Files inside __pycache__, .git, and venv directories should be skipped."""
    # Excluded directories with files inside
    pycache_dir = tmp_path / "__pycache__"
    pycache_dir.mkdir()
    (pycache_dir / "module.cpython-310.pyc").write_bytes(b"\x00\x00")

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config", encoding="utf-8")

    venv_dir = tmp_path / "venv" / "lib"
    venv_dir.mkdir(parents=True)
    (venv_dir / "site.py").write_text("venv stuff", encoding="utf-8")

    # One valid file that SHOULD be found
    (tmp_path / "app.py").write_text("print('hello')", encoding="utf-8")

    docs = discover_files(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "app.py"


def test_skips_deeply_nested_excluded_directory(tmp_path):
    """Excluded directories should be skipped even when deeply nested."""
    deep_cache = tmp_path / "src" / "__pycache__"
    deep_cache.mkdir(parents=True)
    (deep_cache / "utils.cpython-310.pyc").write_bytes(b"\x00")

    (tmp_path / "src" / "utils.py").write_text("x = 1", encoding="utf-8")

    docs = discover_files(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "src/utils.py"


def test_skips_unsupported_extensions(tmp_path):
    """Files with extensions not in the supported set should be skipped."""
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "data.csv").write_text("a,b,c", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")

    docs = discover_files(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "app.py"


def test_skips_files_with_encoding_errors(tmp_path):
    """Files that can't be decoded as UTF-8 should be silently skipped."""
    bad_file = tmp_path / "bad.py"
    bad_file.write_bytes(b"\x80\x81\x82\xff")  # Invalid UTF-8

    good_file = tmp_path / "good.py"
    good_file.write_text("x = 1", encoding="utf-8")

    docs = discover_files(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "good.py"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_directory_returns_empty_list(tmp_path):
    """An empty directory should return an empty list, not an error."""
    docs = discover_files(tmp_path)

    assert docs == []


def test_raises_for_nonexistent_path(tmp_path):
    """A path that doesn't exist should raise ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        discover_files(tmp_path / "nonexistent")


def test_raises_for_file_path(tmp_path):
    """Passing a file (not a directory) should raise ValueError."""
    file_path = tmp_path / "file.txt"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        discover_files(file_path)


# ---------------------------------------------------------------------------
# Metadata correctness
# ---------------------------------------------------------------------------


def test_metadata_has_repo_name(tmp_path):
    """metadata['repo_name'] should be the directory name of the repo root."""
    (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")

    docs = discover_files(tmp_path)

    assert docs[0].metadata["repo_name"] == tmp_path.name


def test_metadata_has_size_bytes(tmp_path):
    """metadata['size_bytes'] should be a positive integer."""
    (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")

    docs = discover_files(tmp_path)

    assert "size_bytes" in docs[0].metadata
    assert docs[0].metadata["size_bytes"] > 0


def test_metadata_has_absolute_path(tmp_path):
    """metadata['absolute_path'] should be the full filesystem path as a string."""
    (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")

    docs = discover_files(tmp_path)

    abs_path = docs[0].metadata["absolute_path"]
    assert isinstance(abs_path, str)
    assert abs_path.endswith("app.py")


def test_metadata_has_all_expected_keys(tmp_path):
    """Every Document should have all five standard metadata keys."""
    (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")

    docs = discover_files(tmp_path)

    expected_keys = {"source", "absolute_path", "extension", "repo_name", "size_bytes"}
    assert set(docs[0].metadata.keys()) == expected_keys
