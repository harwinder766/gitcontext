from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import logfire

from gitcontext.vectorstore.supabase_store import SupabaseVectorStore

class HybridRetriever:
    """Combine BM25 and Supabase vector retrieval using RRF."""

    def __init__(
        self,
        documents: list[Document],
        vector_store: SupabaseVectorStore,
        bm25_k: int = 15,
        vector_k: int = 15,
        final_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        if not documents:
            raise ValueError("Documents cannot be empty.")
        if bm25_k <= 0:
            raise ValueError("bm25_k must be greater than 0.")
        if vector_k <= 0:
            raise ValueError("vector_k must be greater than 0.")
        if final_k <= 0:
            raise ValueError("final_k must be greater than 0.")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0.")

        self._documents = list(documents)
        self._bm25_k = bm25_k
        self._bm25 = BM25Retriever.from_documents(
            self._documents,
            k=bm25_k,
        )

        self._vector_store = vector_store
        self._vector_k = vector_k
        self._final_k = final_k
        self._rrf_k = rrf_k

    @logfire.instrument("Hybrid retrieval")
    def retrieve(
        self,
        query: str,
        filter: dict | None = None,
    ) -> list[Document]:
        """Retrieve documents using BM25 and Supabase vector search."""
        if not query.strip():
            raise ValueError("Query must be a non-empty string.")

        with logfire.span("BM25 retrieval",filter_applied=bool(filter)):
            bm25_results = self._bm25_results(
                query,
                filter,
            )

        with logfire.span("Vector retrieval",top_k=self._vector_k,filter_applied=bool(filter)):
            vector_results = self._vector_store.similarity_search(
                query,
                k=self._vector_k,
                filter=filter,
            )

        with logfire.span("Reciprocal Rank Fusion"):
            results = self._rrf_fusion(
                bm25_results,
                vector_results,
            )

        logfire.debug(
            "Hybrid retrieval completed",
            bm25_results_count=len(bm25_results),
            vector_results_count=len(vector_results),
            final_results_count=len(results),
        )

        return results

    def _bm25_results(
        self,
        query: str,
        filter: dict | None,
    ) -> list[Document]:
        """Run BM25 over only documents matching the requested filter."""
        if not filter:
            return self._bm25.invoke(query)

        filtered_documents = [
            document
            for document in self._documents
            if self._matches_filter(document, filter)
        ]

        if not filtered_documents:
            return []

        filtered_bm25 = BM25Retriever.from_documents(
            filtered_documents,
            k=min(self._bm25_k, len(filtered_documents)),
        )
        return filtered_bm25.invoke(query)

    @staticmethod
    def _matches_filter(
        document: Document,
        filter: dict,
    ) -> bool:
        """Match the equality filters supported by the application."""
        for key, expected_value in filter.items():
            if isinstance(expected_value, dict):
                raise ValueError(
                    "BM25 filtering currently supports equality filters only."
                )

            if document.metadata.get(key) != expected_value:
                return False

        return True

    def _rrf_fusion(
        self,
        bm25_results: list[Document],
        vector_results: list[Document],
    ) -> list[Document]:
        scores: dict[str, float] = {}
        documents: dict[str, Document] = {}

        self._add_ranked_results(
            bm25_results,
            scores,
            documents,
        )
        self._add_ranked_results(
            vector_results,
            scores,
            documents,
        )

        ranked_ids = sorted(
            scores,
            key=lambda document_id: scores[document_id],
            reverse=True,
        )

        results: list[Document] = []

        for document_id in ranked_ids[: self._final_k]:
            document = documents[document_id]
            metadata = dict(document.metadata)
            metadata["rrf_score"] = scores[document_id]

            results.append(
                Document(
                    id=document.id,
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

        return results

    def _add_ranked_results(
        self,
        ranked_documents: list[Document],
        scores: dict[str, float],
        documents: dict[str, Document],
    ) -> None:
        """Accumulate one contribution per unique document in a result list."""
        seen_ids: set[str] = set()

        for rank, document in enumerate(
            ranked_documents,
            start=1,
        ):
            document_id = self._get_document_id(document)

            if document_id in seen_ids:
                continue

            seen_ids.add(document_id)

            scores[document_id] = (
                scores.get(document_id, 0.0)
                + 1.0 / (self._rrf_k + rank)
            )

            documents.setdefault(document_id, document)

    @staticmethod
    def _get_document_id(document: Document) -> str:
        metadata = document.metadata

        required_fields = (
            "repo_name",
            "source",
            "chunk_index",
        )

        missing_fields = [
            field
            for field in required_fields
            if field not in metadata
        ]

        if missing_fields:
            raise ValueError(
                "Document is missing chunk identity metadata: "
                + ", ".join(missing_fields)
            )

        return "|".join(
            str(metadata[field])
            for field in required_fields
        )
    