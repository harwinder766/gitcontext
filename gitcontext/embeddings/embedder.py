"""Generate embeddings for LangChain documents."""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
import logfire
from gitcontext.config import get_env
from sentence_transformers import CrossEncoder

def create_embedding_model() -> Embeddings:
    """Create the configured LangChain-compatible embedding model."""
    embedding_provider = (
        get_env("EMBEDDING_PROVIDER", "huggingface").strip().lower()
    )
    embedding_model = get_env(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    ).strip()

    with logfire.span(
        "Create embedding model: {provider}/{model}",
        provider=embedding_provider,
        model=embedding_model,
    ):
        if embedding_provider != "huggingface":
            raise ValueError(
                f"Unsupported embedding provider: '{embedding_provider}'. "
                "Supported providers: ['huggingface']"
            )

        try:
            return HuggingFaceEmbeddings(model_name=embedding_model)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create embedding model '{embedding_model}': {e}"
            ) from e
        

def embed_documents(
    documents: list[Document],
    embedding_model: Embeddings,
) -> list[list[float]]:
    """Embed each document's page content using the supplied model."""
    if not documents:
        return []

    texts = [document.page_content for document in documents]
    return embedding_model.embed_documents(texts)


def load_cross_encoder_model() -> CrossEncoder:
    """Load a CrossEncoder model for reranking."""
    model_name = get_env(
        "CROSSENCODER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ).strip().lower()
    with logfire.span(
        "Load CrossEncoder model: {model}",
        model=model_name,
    ):
        try:
            return CrossEncoder(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CrossEncoder model '{model_name}': {e}"
            ) from e
    
