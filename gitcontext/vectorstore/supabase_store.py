import os

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.documents import Document
from sqlalchemy import text
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

    def load_documents(self) -> list[Document]:
        """Load all stored chunk documents from this PGVector collection."""
        with self.store._make_sync_session() as session:
            collection = self.store.get_collection(session)

            if collection is None:
                return []

            rows = (
                session.query(self.store.EmbeddingStore)
                .filter(
                    self.store.EmbeddingStore.collection_id
                    == collection.uuid
                )
                .order_by(self.store.EmbeddingStore.id)
                .all()
            )

        return [
            Document(
                id=str(row.id),
                page_content=row.document or "",
                metadata=dict(row.cmetadata or {}),
            )
            for row in rows
        ]


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

    def delete_repository(self, repo_name: str) -> int:
        """Delete all indexed chunks belonging to a repository."""
    
        if not repo_name.strip():
            raise ValueError("Repository name cannot be empty.")
    
        with self.store._make_sync_session() as session:
            result = session.execute(
                text("""
                    DELETE FROM langchain_pg_embedding AS e
                    USING langchain_pg_collection AS c
                    WHERE e.collection_id = c.uuid
                      AND c.name = :collection_name
                      AND e.cmetadata->>'repo_name' = :repo_name
                """),
                {
                    "collection_name": "gitcontext_documents",
                    "repo_name": repo_name,
                },
            )
            deleted_count = result.rowcount
            session.commit()
    
        return deleted_count

    
    