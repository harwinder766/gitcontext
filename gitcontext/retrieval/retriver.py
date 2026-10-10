from langchain_core.documents import Document 

from gitcontext.embeddings.embedder import create_embedding_model
from gitcontext.vectorstore.supabase_store import SupabaseVectorStore   

class Retriever:
    def __init__(self,k = 4):
        """Retrieve relavant documents from Supabase based on a query."""
        if k < 1:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.embedding_model = create_embedding_model()
        self.vector_store = SupabaseVectorStore(self.embedding_model)

    def retrieve(self, query: str, filter : dict | None = None) -> list[Document]:
        """Retrieve relevant documents from Supabase based on a query."""
        if not query.strip():
            raise ValueError("Query must be a non-empty string.")

        return self.vector_store.similarity_search(query, k=self.k, filter=filter)

