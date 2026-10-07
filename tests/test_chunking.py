"""Tests for splitting LangChain documents into smaller documents."""

from langchain_core.documents import Document

from gitcontext.chunking import chunk_documents


CHUNK_SIZE = 20
CHUNK_OVERLAP = 5


def test_splits_long_document_into_multiple_chunks():
    document = Document(page_content="0123456789" * 5)

    chunks = chunk_documents(
        [document],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    assert len(chunks) > 1
    assert all(isinstance(chunk, Document) for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_preserves_content_with_overlap():
    content = "0123456789" * 5
    chunks = chunk_documents(
        [Document(page_content=content)],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    reconstructed = chunks[0].page_content
    for previous, current in zip(chunks, chunks[1:]):
        assert current.page_content.startswith(previous.page_content[-CHUNK_OVERLAP:])
        reconstructed += current.page_content[CHUNK_OVERLAP:]

    assert reconstructed == content


def test_preserves_metadata_in_every_chunk():
    metadata = {
        "source": "src/example.py",
        "absolute_path": "/repo/src/example.py",
        "extension": ".py",
        "repo_name": "example",
        "size_bytes": 50,
    }

    chunks = chunk_documents(
        [Document(page_content="0123456789" * 5, metadata=metadata)],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for chunk in chunks:
        assert chunk.metadata["source"] == metadata["source"]
        assert chunk.metadata["absolute_path"] == metadata["absolute_path"]
        assert chunk.metadata["extension"] == metadata["extension"]
        assert chunk.metadata["repo_name"] == metadata["repo_name"]
        assert chunk.metadata["size_bytes"] == metadata["size_bytes"]


def test_handles_short_document_without_splitting():
    document = Document(
        page_content="short document",
        metadata={"source": "README.md"},
    )

    chunks = chunk_documents(
        [document],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    assert len(chunks) == 1
    assert chunks[0].page_content == document.page_content
    assert chunks[0].metadata["source"] == "README.md"
    assert chunks[0].metadata["chunk_index"] == 0


def test_handles_empty_input():
    assert chunk_documents([]) == []
