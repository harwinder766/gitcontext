from copy import deepcopy
import logfire
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved documents according to query relevance."""

    def __init__(
        self,
        model : CrossEncoder,
        top_n: int = 5,
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be greater than zero")

        self.top_n = top_n
        self.model = model

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

        pairs = [
            (query, document.page_content)
            for document in documents
        ]
        with logfire.span("Score documents with CrossEncoder",document_count=len(pairs)):
            scores = self.model.predict(pairs)

        ranked_results = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        results = []

        for document, score in ranked_results[:self.top_n]:
            reranked_document = Document(
                id=document.id,
                page_content=document.page_content,
                metadata=deepcopy(document.metadata),
            )

            reranked_document.metadata["rerank_score"] = float(score)
            results.append(reranked_document)

        logfire.debug(
            "Document reranking completed",
            input_document_count=len(documents),
            output_document_count=len(results),
        )

        return results
