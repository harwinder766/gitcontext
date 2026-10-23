from __future__ import annotations

from copy import deepcopy
import os
from typing import Any

import httpx
from langchain_core.documents import Document
import logfire


class CrossEncoderReranker:
    """Rerank retrieved documents using the zero-RAM Jina Reranker API."""

    def __init__(
        self,
        model: Any = None,
        top_n: int = 3,
        api_key: str | None = None,
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero")

        self.top_n = top_n
        # Supports reading key passed directly or reading from environment
        self.api_key = (api_key or os.getenv("JINA_API_KEY", "")).strip()
        self.endpoint = "https://api.jina.ai/v1/rerank"

    def _fallback_top_n(self, documents: list[Document]) -> list[Document]:
        """Preserves top_n documents in original order when API is unavailable."""
        results = []
        for doc in documents[: self.top_n]:
            fallback_doc = Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata=deepcopy(doc.metadata),
            )
            fallback_doc.metadata["rerank_score"] = 1.0
            results.append(fallback_doc)
        return results

    @logfire.instrument("Rerank retrieved documents")
    def rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:
        """Return the most relevant documents in ranked order."""
        if not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not documents:
            return []

        # Graceful fallback if JINA_API_KEY is not configured
        if not self.api_key:
            logfire.warn(
                "JINA_API_KEY not provided; preserving original retrieval order.",
                top_n=self.top_n,
            )
            return self._fallback_top_n(documents)

        payload = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [doc.page_content for doc in documents],
            "top_n": min(self.top_n, len(documents)),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with logfire.span(
                "Score documents with Jina Reranker API",
                document_count=len(documents),
                top_n=self.top_n,
            ):
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(self.endpoint, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()

            results: list[Document] = []
            for item in data.get("results", []):
                idx = int(item["index"])
                score = float(item["relevance_score"])
                if 0 <= idx < len(documents):
                    doc = documents[idx]
                    reranked_doc = Document(
                        id=doc.id,
                        page_content=doc.page_content,
                        metadata=deepcopy(doc.metadata),
                    )
                    reranked_doc.metadata["rerank_score"] = score
                    results.append(reranked_doc)

            logfire.debug(
                "Document reranking completed via Jina API",
                input_document_count=len(documents),
                output_document_count=len(results),
            )
            return results

        except Exception as error:
            logfire.warn(
                "Jina reranker API call failed; falling back to original order",
                error=str(error),
            )
            return self._fallback_top_n(documents)