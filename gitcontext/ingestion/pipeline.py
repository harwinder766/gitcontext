from pathlib import Path
from urllib.parse import urlparse

from gitcontext.chunking.splitter import chunk_documents
from gitcontext.embeddings.embedder import create_embedding_model
from gitcontext.vectorstore.supabase_store import SupabaseVectorStore
from gitcontext.ingestion.discovery import discover_files
from gitcontext.ingestion.commit_ingestion import CommitIngestion
from gitcontext.ingestion.pr_ingestion import PullRequestIngestion
from gitcontext.ingestion.repository_cloner import clone_repository

def extract_github_repo(repo_url: str) -> str:
    parsed_url = urlparse(repo_url)

    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != "github.com"
        or parsed_url.port is not None
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError(
            "Please provide a valid HTTPS GitHub repository URL."
        )

    parts = parsed_url.path.strip("/").split("/")

    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "URL must be in the format "
            "https://github.com/owner/repository."
        )

    repository = parts[1].removesuffix(".git")
    if not repository:
        raise ValueError("Repository name cannot be empty.")

    return f"{parts[0]}/{repository}"


def ingest_repository(repo_url: str) -> None:
    """Ingest a repository by discovering files, chunking them, embedding, and storing in Supabase."""
    if not repo_url:
        raise ValueError("Repository URL cannot be empty.")

    github_repo = extract_github_repo(repo_url)

    # Step 0: Clone the repository to a temporary directory
    repo_path = clone_repository(repo_url)
    
    # Step 1: Discover files in the repository
    documents = discover_files(repo_path, repo_name=github_repo)

    # Step 2: Ingest commits
    commit_ingestion = CommitIngestion(
        repo_path=str(repo_path),
        repo_name=github_repo,
    )
    commit_documents = commit_ingestion.commits_traverse()
    documents.extend(commit_documents)

    # Step 3: Ingest pull requests
    pr_ingestion = PullRequestIngestion(repo_full_name=github_repo)
    pr_documents = pr_ingestion.pulls_traverse()
    documents.extend(pr_documents)

    if not documents:
        print("No relevant documents found in the repository after file ingestion, commit traversal and pull request traversal.")
        return
    
    # Step 4: Chunk the documents
    chunked_documents = chunk_documents(documents)

    # Step 5: Create the embedding model
    embedding_model = create_embedding_model()

    # Step 6: Store the chunked documents in Supabase
    vector_store = SupabaseVectorStore(embedding_model)

    # Remove previously indexed chunks for this repository
    deleted_count = vector_store.delete_repository(github_repo)
    print(f"Removed {deleted_count} previously indexed chunks.")

    ids = vector_store.add_documents(chunked_documents)

    print(f"Successfully ingested {len(ids)} chunks into Supabase.")
