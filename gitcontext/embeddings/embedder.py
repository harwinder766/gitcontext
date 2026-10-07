"""Generate embeddings for LangChain documents."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from gitcontext.config.settings import Settings, get_settings


def create_embedding_model(settings: Settings | None = None) -> Embeddings:
    """Create the configured LangChain-compatible embedding model."""
    resolved_settings = settings or get_settings()

    if resolved_settings.embedding_provider != "huggingface":
        raise ValueError(
            f"Unsupported embedding provider: {resolved_settings.embedding_provider}"
        )

    return HuggingFaceEmbeddings(
        model_name=resolved_settings.embedding_model,
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
