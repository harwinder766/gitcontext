"""Generate embeddings for LangChain documents."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from gitcontext.config import get_env


def create_embedding_model() -> Embeddings:
    """Create the configured LangChain-compatible embedding model."""
    embedding_provider = get_env("EMBEDDING_PROVIDER", "huggingface")
    embedding_model = get_env(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    if embedding_provider != "huggingface":
        raise ValueError(
            f"Unsupported embedding provider: {embedding_provider}"
        )

    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_documents(
    documents: list[Document],
    embedding_model: Embeddings,
) -> list[list[float]]:
    """Embed each document's page content using the supplied model."""
    if not documents:
        return []

    texts = [document.page_content for document in documents]
    return embedding_model.embed_documents(texts)
