# from __future__ import annotations

# from langchain_core.documents import Document
# from langchain_community.retrievers import BM25Retriever


# class HybridRetriever:
#     """
#     Hybrid retriever combining BM25 keyword retrieval
#     with semantic vector retrieval using RRF.
#     """

#     def __init__(
#         self,
#         documents: list[Document],
#         vector_store,
#         bm25_k: int = 10,
#         vector_k: int = 10,
#         final_k: int = 5,
#         rrf_k: int = 60,
#     ) -> None:

#         if not documents:
#             raise ValueError(
#                 "Documents cannot be empty."
#             )

#         if bm25_k <= 0:
#             raise ValueError(
#                 "bm25_k must be greater than 0."
#             )

#         if vector_k <= 0:
#             raise ValueError(
#                 "vector_k must be greater than 0."
#             )

#         if final_k <= 0:
#             raise ValueError(
#                 "final_k must be greater than 0."
#             )

#         if rrf_k <= 0:
#             raise ValueError(
#                 "rrf_k must be greater than 0."
#             )

#         # BM25 retriever
#         self._bm25 = BM25Retriever.from_documents(
#             documents,
#             k=bm25_k,
#         )

#         # Your custom ChromaStore
#         self._vector_store = vector_store
#         self._vector_k = vector_k

#         self._final_k = final_k
#         self._rrf_k = rrf_k

#     def invoke(
#         self,
#         query: str,
#     ) -> list[Document]:

#         if not query.strip():
#             raise ValueError(
#                 "Query cannot be empty."
#             )

#         # -----------------------------------------
#         # BM25 retrieval
#         # -----------------------------------------

#         bm25_results = self._bm25.invoke(
#             query
#         )

#         # -----------------------------------------
#         # Vector retrieval
#         # -----------------------------------------

#         vector_results = self._vector_store.search(
#             query,
#             k=self._vector_k,
#         )

#         # -----------------------------------------
#         # RRF fusion
#         # -----------------------------------------

#         return self._rrf_fusion(
#             bm25_results,
#             vector_results,
#         )

#     def _rrf_fusion(
#         self,
#         bm25_results: list[Document],
#         vector_results: list[Document],
#     ) -> list[Document]:

#         scores: dict[str, float] = {}
#         documents: dict[str, Document] = {}

#         # =========================================
#         # BM25 results
#         # =========================================

#         for rank, document in enumerate(
#             bm25_results,
#             start=1,
#         ):

#             document_id = self._get_document_id(
#                 document
#             )

#             scores[document_id] = (
#                 scores.get(document_id, 0.0)
#                 + 1.0
#                 / (self._rrf_k + rank)
#             )

#             documents[document_id] = document

#         # =========================================
#         # Vector results
#         # =========================================

#         for rank, document in enumerate(
#             vector_results,
#             start=1,
#         ):

#             document_id = self._get_document_id(
#                 document
#             )

#             scores[document_id] = (
#                 scores.get(document_id, 0.0)
#                 + 1.0
#                 / (self._rrf_k + rank)
#             )

#             documents[document_id] = document

#         # =========================================
#         # Sort by RRF score
#         # =========================================

#         ranked_ids = sorted(
#             scores,
#             key=scores.get,
#             reverse=True,
#         )

#         # =========================================
#         # Return top final_k documents
#         # =========================================

#         results: list[Document] = []

#         for document_id in ranked_ids[
#             :self._final_k
#         ]:

#             document = documents[document_id]

#             document.metadata["rrf_score"] = (
#                 scores[document_id]
#             )

#             results.append(document)

#         return results

#     @staticmethod
#     def _get_document_id(
#         document: Document,
#     ) -> str:

#         return document.metadata.get(
#             "chunk_id",
#             document.page_content,
#         )

# from collections import defaultdict

# file_indices = defaultdict(int)

# file_indices["file1.py"] += 1
# file_indices["file2.py"] += 1
# file_indices["file1.py"] += 1  # Incrementing the count for file1.py

# print("File indices:", file_indices)

from langchain_community.retrievers import BM25Retriever
