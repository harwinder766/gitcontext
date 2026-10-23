"""Generate embeddings for LangChain documents with 0 MB local RAM footprint."""

from __future__ import annotations

from typing import Any
import httpx
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import logfire

from gitcontext.config import get_env


# class JinaEmbedding(Embeddings):
#     """Zero-RAM embedding provider using Jina Embeddings v3 API (384 dimensions)."""

#     def __init__(self, api_key: str, model: str = "jina-embeddings-v3") -> None:
#         self.api_key = api_key
#         self.model = model
#         self.endpoint = "https://api.jina.ai/v1/embeddings"
#         self.headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#         }

#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         if not texts:
#             return []

#         all_embeddings: list[list[float]] = []
#         batch_size = 64

#         try:
#             with httpx.Client(timeout=30.0) as client:
#                 for i in range(0, len(texts), batch_size):
#                     batch = texts[i : i + batch_size]
#                     payload = {
#                         "model": self.model,
#                         "task": "retrieval.passage",
#                         "dimensions": 384,
#                         "input": batch,
#                     }
#                     resp = client.post(
#                         self.endpoint,
#                         json=payload,
#                         headers=self.headers,
#                     )
#                     resp.raise_for_status()

#                     data = sorted(
#                         resp.json()["data"],
#                         key=lambda item: item["index"],
#                     )
#                     batch_embeddings = [item["embedding"] for item in data]

#                     if len(batch_embeddings) != len(batch):
#                         raise ValueError(
#                             "The embedding API returned an unexpected number of embeddings."
#                         )
#                     all_embeddings.extend(batch_embeddings)

#             return all_embeddings

#         except (httpx.HTTPStatusError, httpx.RequestError):
#             raise
#         except (KeyError, IndexError, TypeError) as exc:
#             raise ValueError(
#                 "Invalid response structure from the Jina embedding API."
#             ) from exc

#     def embed_query(self, text: str) -> list[float]:
#         if not text or not text.strip():
#             raise ValueError("Query text cannot be empty or whitespace-only.")

#         try:
#             payload = {
#                 "model": self.model,
#                 "task": "retrieval.query",
#                 "dimensions": 384,
#                 "input": [text],
#             }

#             with httpx.Client(timeout=30.0) as client:
#                 resp = client.post(
#                     self.endpoint,
#                     json=payload,
#                     headers=self.headers,
#                 )
#                 resp.raise_for_status()

#                 data = resp.json()["data"]
#                 embedding = data[0]["embedding"]

#                 if len(embedding) != 384:
#                     raise ValueError(
#                         f"Expected 384 dimensions, got {len(embedding)}."
#                     )

#                 return embedding

#         except (httpx.HTTPStatusError, httpx.RequestError):
#             raise
#         except (KeyError, IndexError, TypeError) as exc:
#             raise ValueError(
#                 "Invalid response structure from the Jina embedding API."
#             ) from exc


import time
import httpx
from langchain_core.embeddings import Embeddings
import logfire


class JinaEmbedding(Embeddings):
    """Zero-RAM embedding provider using Jina Embeddings v3 API (384 dimensions) with rate-limit backoff."""

    def __init__(self, api_key: str, model: str = "jina-embeddings-v3") -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_with_retry(
        self,
        client: httpx.Client,
        payload: dict,
        max_retries: int = 5,
    ) -> dict:
        """Send POST request to Jina API with automatic exponential backoff on 429 rate limits."""
        for attempt in range(max_retries):
            resp = client.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
            )
            if resp.status_code == 429:
                # Check if Jina sent a Retry-After header, otherwise use exponential backoff (5s, 10s, 20s...)
                retry_after = float(resp.headers.get("Retry-After", 5 * (attempt + 1)))
                logfire.warn(
                    "Jina API 429 rate limit hit (attempt {attempt}/{max_retries}). Sleeping {wait}s...",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    wait=retry_after,
                )
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            return resp.json()

        # Final check if all retries were exhausted
        resp.raise_for_status()
        return resp.json()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        batch_size = 32  # Smaller batch size prevents single-request TPM spikes

        try:
            with httpx.Client(timeout=60.0) as client:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    payload = {
                        "model": self.model,
                        "task": "retrieval.passage",
                        "dimensions": 384,
                        "input": batch,
                    }
                    resp_json = self._post_with_retry(client, payload)

                    data = sorted(
                        resp_json["data"],
                        key=lambda item: item["index"],
                    )
                    batch_embeddings = [item["embedding"] for item in data]

                    if len(batch_embeddings) != len(batch):
                        raise ValueError(
                            "The embedding API returned an unexpected number of embeddings."
                        )
                    all_embeddings.extend(batch_embeddings)

                    # Brief pause between batches if there are more chunks to process
                    if i + batch_size < len(texts):
                        time.sleep(1.0)

            return all_embeddings

        except (httpx.HTTPStatusError, httpx.RequestError):
            raise
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Invalid response structure from the Jina embedding API."
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty or whitespace-only.")

        try:
            payload = {
                "model": self.model,
                "task": "retrieval.query",
                "dimensions": 384,
                "input": [text],
            }

            with httpx.Client(timeout=30.0) as client:
                resp_json = self._post_with_retry(client, payload)
                data = resp_json["data"]
                embedding = data[0]["embedding"]

                if len(embedding) != 384:
                    raise ValueError(
                        f"Expected 384 dimensions, got {len(embedding)}."
                    )

                return embedding

        except (httpx.HTTPStatusError, httpx.RequestError):
            raise
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(
                "Invalid response structure from the Jina embedding API."
            ) from exc

class GeminiEmbedding(Embeddings):
    """Zero-RAM embedding provider using Google Gemini gemini-embedding-001 (384 dimensions)."""

    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001") -> None:
        self.api_key = api_key
        clean_model = model.replace("models/", "").strip()
        self.model = f"models/{clean_model}"
        self.endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/{self.model}:batchEmbedContents"
        )
        self.headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        batch_size = 64

        with httpx.Client(timeout=30.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                requests = [
                    {
                        "model": self.model,
                        "content": {"parts": [{"text": text}]},
                        "outputDimensionality": 384,
                    }
                    for text in batch
                ]
                resp = client.post(
                    self.endpoint,
                    params={"key": self.api_key},
                    json={"requests": requests},
                    headers=self.headers,
                )
                resp.raise_for_status()
                embeddings = resp.json().get("embeddings", [])
                results.extend(
                    item.get("values", item.get("embedding", [])) for item in embeddings
                )

        return results

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty or whitespace-only.")
        embeddings = self.embed_documents([text])
        return embeddings[0]


def create_embedding_model() -> Embeddings:
    """Create a LangChain-compatible embedding model outputting 384-dim vectors."""
    provider = get_env("EMBEDDING_PROVIDER", "").strip().lower()
    jina_key = get_env("JINA_API_KEY", "").strip()
    gemini_key = get_env("GEMINI_API_KEY", "").strip()

    if not jina_key and not gemini_key:
        raise ValueError(
            "No embedding API key found. Please set JINA_API_KEY or GEMINI_API_KEY in your .env file."
        )

    with logfire.span("Create 384-dim embedding model"):
        if provider in ("google", "gemini") and gemini_key:
            return GeminiEmbedding(api_key=gemini_key)

        if jina_key:
            return JinaEmbedding(api_key=jina_key)

        return GeminiEmbedding(api_key=gemini_key)


def embed_documents(
    documents: list[Document],
    embedding_model: Embeddings,
) -> list[list[float]]:
    """Embed each document's page content using the supplied model."""
    if not documents:
        return []

    texts = [document.page_content for document in documents]
    return embedding_model.embed_documents(texts)


def load_cross_encoder_model() -> Any:
    """Load a lightweight reranker instance for application startup."""
    from gitcontext.reranking.reranker import CrossEncoderReranker

    return CrossEncoderReranker(top_n=3)