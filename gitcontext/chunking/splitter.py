"""Split LangChain documents into smaller documents for downstream processing."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

COMMIT_DIFF_MARKER = "\n\nDiff:\n"
MAX_COMMIT_MSG_CHARS = 200


def _chunk_commit_document(
    document: Document, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split only the diff, and put a short identity header on every chunk."""
    meta = document.metadata
    _, sep, diff = document.page_content.partition(COMMIT_DIFF_MARKER)
    if not sep:  # unexpected format, fall back to normal splitting
        diff = document.page_content

    header = (
        f"Commit hash: {meta.get('commit_hash')}\n"
        f"Commit message: {(meta.get('commit_msg') or '')[:MAX_COMMIT_MSG_CHARS].strip()}\n"
        f"File path: {meta.get('source') or meta.get('file_path')}\n"
        f"Change type: {meta.get('change_type')}"
    )

    # leave room for the header so each chunk stays near chunk_size
    body_size = max(200, chunk_size - len(header) - 30)
    body_overlap = min(chunk_overlap, body_size // 4)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=body_size, chunk_overlap=body_overlap
    )
    parts = splitter.split_text(diff)

    return [
        Document(
            page_content=f"{header}\n\nDiff (part {i + 1}/{len(parts)}):\n{part}",
            metadata=dict(meta),
        )
        for i, part in enumerate(parts)
    ]


def _chunk_pr_document(
    document: Document, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split PR content while repeating its identity on every chunk."""
    meta = document.metadata

    header = (
        f"PR number: {meta.get('pr_number')}\n"
        f"PR title: {meta.get('pr_title')}\n"
        f"PR state: {meta.get('pr_state')}\n"
        f"Item type: {meta.get('pr_item_type')}\n"
        f"Author: {meta.get('author_name')}\n"
        f"Source: {meta.get('source')}"
    )

    # Preserve the original content while removing its duplicated header.
    marker = "\n\nBody:\n"
    _, sep, body = document.page_content.partition(marker)

    if not sep:
        body = document.page_content

    body_size = max(200, chunk_size - len(header) - 40)
    body_overlap = min(chunk_overlap, body_size // 4)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=body_size,
        chunk_overlap=body_overlap,
    )

    parts = splitter.split_text(body)

    return [
        Document(
            page_content=(
                f"{header}\n\n"
                f"Content (part {i + 1}/{len(parts)}):\n{part}"
            ),
            metadata=dict(meta),
        )
        for i, part in enumerate(parts)
    ]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: list[Document] = []

    for document in documents:
        if document.metadata.get("document_type") == "commit":
            document_chunks = _chunk_commit_document(document, chunk_size, chunk_overlap)
        elif document.metadata.get("document_type") == "pull_request":
            document_chunks = _chunk_pr_document(document, chunk_size, chunk_overlap)
        else:
            document_chunks = splitter.split_documents([document])

        for chunk_index, chunk in enumerate(document_chunks):
            chunk.metadata["chunk_index"] = chunk_index
        chunks.extend(document_chunks)

    return chunks