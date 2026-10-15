from pathlib import Path
from urllib.parse import urlparse
from langchain_core.embeddings import Embeddings

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


import os
import shutil
import stat


def _remove_readonly(func, path, _):
    """Windows keeps .git files read-only, so make them writable before deleting."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def ingest_repository(
    repo_url: str,
    max_prs: int = 50,
    max_commits: int | None = None,
    embedding_model: Embeddings | None = None,
) -> dict:
    """Ingest a repository: files, commits and pull requests."""
    if not repo_url:
        raise ValueError("Repository URL cannot be empty.")
    if max_prs < 0:
        raise ValueError("max_prs must be zero or a positive integer.")
    if max_commits is not None and max_commits < 0:
        raise ValueError("max_commits must be zero or a positive integer.")

    github_repo = extract_github_repo(repo_url)
    repo_path = clone_repository(repo_url)

    try:
        # Step 1: files
        documents = discover_files(repo_path, repo_name=github_repo)

        # Step 2: commits
        commit_documents = CommitIngestion(
            repo_path=str(repo_path),
            repo_name=github_repo,
            max_commits=max_commits,
        ).commits_traverse()
        documents.extend(commit_documents)

        # Step 3: pull requests (optional, a failure here is not fatal)
        pr_documents = []
        if max_prs > 0:
            try:
                pr_documents = PullRequestIngestion(
                    repo_full_name=github_repo,
                    max_prs=max_prs,
                ).pulls_traverse()
            except Exception as error:
                print(f"Skipping pull requests: {error}")
        documents.extend(pr_documents)
    finally:
        shutil.rmtree(repo_path, onerror=_remove_readonly)   # always clean the temp clone

    if not documents:
        raise ValueError("No relevant documents found in the repository.")

    # Step 4: chunk
    chunked_documents = chunk_documents(documents)

    # Step 5: embedding model (the API passes one in, so it loads only once)
    embedding_model = embedding_model or create_embedding_model()

    # Step 6: store (remove the old chunks of this repo first)
    vector_store = SupabaseVectorStore(embedding_model)
    deleted_count = vector_store.delete_repository(github_repo)
    ids = vector_store.add_documents(chunked_documents)

    return {
        "repo_name": github_repo,
        "commit_documents": len(commit_documents),
        "pr_documents": len(pr_documents),
        "chunks": len(ids),
        "removed_old_chunks": deleted_count,
    }
