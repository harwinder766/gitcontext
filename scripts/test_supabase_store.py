from langchain_core.documents import Document

from gitcontext.embeddings.embedder import create_embedding_model
from gitcontext.vectorstore.supabase_store import SupabaseVectorStore


def main():
    # Use the same embedding model already used by GitContext
    embedding_model = create_embedding_model()

    # Create two small test documents
    documents = [
        Document(
            page_content="Python is a programming language.",
            metadata={
                "file_path": "test/python.py",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Supabase provides PostgreSQL with pgvector support.",
            metadata={
                "file_path": "test/supabase.md",
                "chunk_index": 0,
            },
        ),
    ]

    # Connect to Supabase through our vector-store wrapper
    vector_store = SupabaseVectorStore(embedding_model)

    # Store the documents and their embeddings
    ids = vector_store.add_documents(documents)

    print(f"Successfully added {len(ids)} documents.")
    print("Document IDs:")
    for document_id in ids:
        print(f"  {document_id}")


if __name__ == "__main__":
    main()