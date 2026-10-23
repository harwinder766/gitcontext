from pathlib import Path
from urllib.parse import urlparse
from langchain_core.embeddings import Embeddings
import logfire
import os
import shutil
import stat

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

    # The repository name is useful trace context; avoid recording the full URL.
    with logfire.span("Ingest repository", repo_name=github_repo):
        repo_path = clone_repository(repo_url)

        try:
            # Step 1: discover source files and documentation.
            documents = discover_files(repo_path, repo_name=github_repo)
            # Step 2: extract commit history.
            commit_documents = CommitIngestion(
                repo_path=str(repo_path),
                repo_name=github_repo,
                max_commits=max_commits,
            ).commits_traverse()
            documents.extend(commit_documents)

            # Step 3: pull requests are optional; failures remain non-fatal.
            pr_documents = []
            if max_prs > 0:
                try:
                    pr_documents = PullRequestIngestion(
                        repo_full_name=github_repo,
                        max_prs=max_prs,
                    ).pulls_traverse()
                except Exception as error:
                    logfire.warn(
                        "Skipping pull requests because ingestion failed",
                        repo_name=github_repo,
                        error=str(error),
                    )
            documents.extend(pr_documents)
        finally:
            # Keep cleanup guaranteed, even if discovery or ingestion fails.
            with logfire.span("Clean up cloned repository"):
                shutil.rmtree(repo_path, onerror=_remove_readonly)

        if not documents:
            raise ValueError("No relevant documents found in the repository.")

        # Step 4: split documents into chunks.
        chunked_documents = chunk_documents(documents)
        
        # Step 5: reuse the supplied embedding model, or create one if absent.
        embedding_model = embedding_model or create_embedding_model()

        # Step 6: replace this repository's existing vectors.
        with logfire.span("Update Supabase vector store", chunk_count=len(chunked_documents)):
            vector_store = SupabaseVectorStore(embedding_model)

            with logfire.span("Delete existing repository chunks"):
                deleted_count = vector_store.delete_repository(github_repo)

            with logfire.span("Insert repository chunks", chunk_count=len(chunked_documents)):
                ids = vector_store.add_documents(chunked_documents)

        result = {
            "repo_name": github_repo,
            "commit_documents": len(commit_documents),
            "pr_documents": len(pr_documents),
            "chunks": len(ids),
            "removed_old_chunks": deleted_count,
        }
        logfire.info(
            "Repository ingestion completed",
            repo_name=github_repo,
            commit_document_count=result["commit_documents"],
            pr_document_count=result["pr_documents"],
            chunk_count=result["chunks"],
            removed_old_chunk_count=result["removed_old_chunks"],
        )
        del chunked_documents
        del documents
        del commit_documents
        del pr_documents
        import gc
        gc.collect()

        return result
