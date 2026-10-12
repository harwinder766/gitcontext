from gitcontext.chunking.splitter import chunk_documents
from gitcontext.embeddings.embedder import create_embedding_model
from gitcontext.vectorstore.supabase_store import SupabaseVectorStore
from gitcontext.ingestion.discovery import discover_files
from gitcontext.ingestion.commit_ingestion import CommitIngestion
from pathlib import Path

def ingest_repository(repo_path: Path) -> None:
    """Ingest a repository by discovering files, chunking them, embedding, and storing in Supabase."""
    # Step 1: Discover files in the repository
    documents = discover_files(repo_path)
    if not documents:
        print("No relevant files found in the repository.")
        return

    commit_ingestion = CommitIngestion(repo_path=str(repo_path))
    commit_documents = commit_ingestion.commits_traverse()
    documents.extend(commit_documents)
    if not documents:
        print("No relevant documents found in the repository after commit traversal.")
        return
    
    # Step 2: Chunk the documents
    chunked_documents = chunk_documents(documents)

    # Step 3: Create the embedding model
    embedding_model = create_embedding_model()

    # Step 4: Store the chunked documents in Supabase
    vector_store = SupabaseVectorStore(embedding_model)

    # Remove previously indexed chunks for this repository
    deleted_count = vector_store.delete_repository(repo_path.name)
    print(f"Removed {deleted_count} previously indexed chunks.")

    ids = vector_store.add_documents(chunked_documents)

    print(f"Successfully ingested {len(ids)} chunks into Supabase.")
