from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import logfire

from gitcontext.embeddings.embedder import create_embedding_model
from gitcontext.retrieval.hybrid_retriever import HybridRetriever
from gitcontext.vectorstore.supabase_store import SupabaseVectorStore


class Retriever:
    """Application-level retriever backed by BM25 and Supabase PGVector."""

    def __init__(
        self,
        k: int = 15,
        bm25_k: int = 15,
        vector_k: int = 15,
        rrf_k: int = 60,
        embedding_model: Embeddings | None = None,
        auto_refresh: bool = True,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if bm25_k <= 0:
            raise ValueError("bm25_k must be a positive integer.")
        if vector_k <= 0:
            raise ValueError("vector_k must be a positive integer.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer.")

        self.k = k
        self._bm25_k = bm25_k
        self._vector_k = vector_k
        self._rrf_k = rrf_k

        self.embedding_model = embedding_model or create_embedding_model()
        self.vector_store = SupabaseVectorStore(self.embedding_model)

        self._hybrid_retriever: HybridRetriever | None = None
        if auto_refresh:
            try:
                self.refresh()
            except ValueError as error:
                logfire.warn(
                    "Initial retriever refresh deferred (database may be empty): {error}",
                    error=str(error),
                )

    @logfire.instrument("Refresh application retriever")
    def refresh(self) -> None:
        """Reload stored chunks and rebuild the in-memory BM25 index."""
        with logfire.span("Loading indexed documents from Supabase"):
            documents = self.vector_store.load_documents()

        if not documents:
            raise ValueError(
                "No indexed documents were found in Supabase."
            )
        with logfire.span("Building hybrid retriever"):
            self._hybrid_retriever = HybridRetriever(
                documents=documents,
                vector_store=self.vector_store,
                bm25_k=self._bm25_k,
                vector_k=self._vector_k,
                final_k=self.k,
                rrf_k=self._rrf_k,
            )

        logfire.info("Application retriever refreshed")

    
    def retrieve(
        self,
        query: str,
        filter: dict | None = None,
    ) -> list[Document]:
        """Retrieve documents using BM25 and Supabase PGVector."""
        if self._hybrid_retriever is None:
            self.refresh()

        return self._hybrid_retriever.retrieve(
            query,
            filter=filter,
        )