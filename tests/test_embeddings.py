"""Tests for embedding LangChain documents."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from gitcontext.embeddings import create_embedding_model, embed_documents

class FakeEmbeddings(Embeddings):
    """Deterministic embedding model for unit tests."""

    def __init__(self, dimensions: int = 3):
        self.dimensions = dimensions
        self.received_texts: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.received_texts.extend(texts)
        return [
            [float(len(text))] + [0.0] * (self.dimensions - 1)
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))] + [0.0] * (self.dimensions - 1)


def test_embeds_single_document():
    model = FakeEmbeddings()

    embeddings = embed_documents([Document(page_content="hello")], model)

    assert embeddings == [[5.0, 0.0, 0.0]]
    assert model.received_texts == ["hello"]


def test_embeds_multiple_documents_with_consistent_dimensions():
    model = FakeEmbeddings(dimensions=4)
    documents = [
        Document(page_content="first"),
        Document(page_content="second"),
    ]

    embeddings = embed_documents(documents, model)

    assert len(embeddings) == 2
    assert all(len(vector) == 4 for vector in embeddings)
    assert model.received_texts == ["first", "second"]


def test_empty_input_returns_empty_embeddings_without_calling_model():
    model = FakeEmbeddings()

    assert embed_documents([], model) == []
    assert model.received_texts == []


def test_embedding_model_errors_are_propagated():
    class FailingEmbeddings(FakeEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("embedding service unavailable")

    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        embed_documents([Document(page_content="hello")], FailingEmbeddings())


def test_create_embedding_model_uses_jina_when_key_set(monkeypatch):
    from gitcontext.embeddings.embedder import JinaEmbedding

    monkeypatch.setenv("JINA_API_KEY", "jina_test_key_123")
    monkeypatch.setenv("GEMINI_API_KEY", "")

    model = create_embedding_model()
    assert isinstance(model, JinaEmbedding)
    assert model.api_key == "jina_test_key_123"


def test_create_embedding_model_uses_gemini_when_specified(monkeypatch):
    from gitcontext.embeddings.embedder import GeminiEmbedding

    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini_test_key_456")

    model = create_embedding_model()
    assert isinstance(model, GeminiEmbedding)
    assert model.api_key == "gemini_test_key_456"


def test_create_embedding_model_raises_without_api_keys(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "")

    with pytest.raises(ValueError, match="No embedding API key found"):
        create_embedding_model()
