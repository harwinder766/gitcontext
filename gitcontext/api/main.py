import os
from datetime import datetime, timezone
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException

from gitcontext.api.schemas import AskRequest, IngestRequest
from gitcontext.embeddings.embedder import create_embedding_model
from gitcontext.generation.generation import RAG
from gitcontext.ingestion.pipeline import extract_github_repo, ingest_repository
from gitcontext.vectorstore.supabase_store import SupabaseVectorStore

app = FastAPI(
    title="GitContext API",
    description=(
        "API for GitContext, a tool for analyzing Git repositories and "
        "extracting contextual information."
    ),
    version="1.0.0",
)

_INGESTION_STATUS: dict[str, dict[str, object]] = {}


def _set_status(repo_name: str, status: str, message: str, error: str | None = None) -> None:
    _INGESTION_STATUS[repo_name] = {
        "repo_name": repo_name,
        "status": status,
        "message": message,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _get_repo_list() -> list[str]:
    database_url = os.getenv("SUPABASE_DATABASE_URL")
    if not database_url:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured.",
        )

    vector_store = SupabaseVectorStore(create_embedding_model())
    documents = vector_store.load_documents()
    repositories = {
        document.metadata.get("repo_name")
        for document in documents
        if document.metadata.get("repo_name")
    }
    return sorted(repositories)


@app.get("/")
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, object]:
    return {
        "status": "ok",
        "vector_store_configured": bool(os.getenv("SUPABASE_DATABASE_URL")),
        "repositories": _get_repo_list() if os.getenv("SUPABASE_DATABASE_URL") else [],
    }


@app.post("/index")
def index(request: IngestRequest) -> dict[str, object]:
    repo_name = extract_github_repo(request.repo_url)
    _set_status(repo_name, "running", "Ingestion started.")

    try:
        result = ingest_repository(
            repo_url=request.repo_url,
            max_commits=request.max_commits,
            max_prs=request.max_prs,
        )
        _set_status(repo_name, "done", "Ingestion completed.", error=None)
        return {
            "status": "done",
            "repo_name": repo_name,
            "result": result,
        }
    except ValueError as error:
        _set_status(repo_name, "failed", "Ingestion failed.", str(error))
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        _set_status(repo_name, "failed", "Ingestion failed.", "Repository ingestion failed.")
        raise HTTPException(status_code=500, detail="Repository ingestion failed.") from error


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict[str, object]:
    """Backward-compatible alias for repository indexing."""
    return index(request)


@app.get("/ingest/{repo_name:path}")
def get_ingest_status(repo_name: str) -> dict[str, object]:
    repo_name = unquote(repo_name)
    status_data = _INGESTION_STATUS.get(repo_name)
    if status_data is None:
        return {
            "repo_name": repo_name,
            "status": "unknown",
            "message": "No ingestion status has been recorded for this repository.",
            "error": None,
        }
    return status_data


@app.post("/query")
def query(request: AskRequest) -> object:
    """Query a GitHub repository."""
    repo_name = request.repo_name.strip()
    if not repo_name:
        raise HTTPException(status_code=400, detail="repo_name is required.")

    try:
        indexed_repositories = _get_repo_list()
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=503, detail="Repository index is unavailable.") from error

    if repo_name not in indexed_repositories:
        raise HTTPException(
            status_code=404,
            detail=f"Repository '{repo_name}' is not indexed.",
        )

    try:
        rag = RAG(
            query=request.question,
            filter={"repo_name": repo_name},
        )
        return rag.generate_response()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Query failed.") from error


@app.post("/ask")
def ask(request: AskRequest) -> object:
    """Backward-compatible alias for repository-aware querying."""
    return query(request)


@app.get("/repositories")
def list_repositories() -> dict[str, list[str]]:
    try:
        repositories = _get_repo_list()
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=503, detail="Repository index is unavailable.") from error

    return {"repositories": repositories}


@app.delete("/repositories/{repo_name:path}")
def delete_repository(repo_name: str) -> dict[str, object]:
    repo_name = unquote(repo_name)
    try:
        vector_store = SupabaseVectorStore(create_embedding_model())
        deleted_count = vector_store.delete_repository(repo_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Repository deletion failed.") from error

    _INGESTION_STATUS.pop(repo_name, None)
    return {
        "repo_name": repo_name,
        "deleted_chunks": deleted_count,
        "status": "deleted",
    }
