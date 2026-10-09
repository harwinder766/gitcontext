import os

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.documents import Document
load_dotenv()


class SupabaseVectorStore:
    def __init__(self, embedding_model):
        database_url = os.getenv("SUPABASE_DATABASE_URL")

        if not database_url:
            raise ValueError(
                "SUPABASE_DATABASE_URL is not set in the environment."
            )

        self.store = PGVector(
            embeddings=embedding_model,
            collection_name="gitcontext_documents",
            connection=database_url,
            use_jsonb=True,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        '''Add documents to Supabase.'''
        return self.store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4, filter : dict | None = None) -> list[Document]:
        '''Perform a similarity search in Supabase with optional filtering.'''
        if not query:
            raise ValueError("Query string cannot be empty.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        return self.store.similarity_search(query, k=k, filter=filter)

    def delete_documents(self, ids: list[str]) -> None:
        '''Delete documents from Supabase by their IDs.'''
        if not ids:
            raise ValueError("ids list cannot be empty.")
        self.store.delete_documents(ids=ids)

    