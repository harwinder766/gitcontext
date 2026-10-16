import os
from uuid import uuid4
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException

from gitcontext.api.schemas import AskRequest, IngestRequest
from gitcontext.embeddings.embedder import create_embedding_model
from gitcontext.generation.llm import create_llm
from gitcontext.graph.graph import build_graph
from gitcontext.ingestion.pipeline import extract_github_repo, ingest_repository
from gitcontext.vectorstore.supabase_store import SupabaseVectorStore

_INGESTION_STATUS: dict[str, dict[str, object]] = {}


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize process-wide services once for the API lifetime."""
    application.state.embedding_model = create_embedding_model()
    application.state.llm = create_llm()
    database_url = os.getenv("SUPABASE_DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "SUPABASE_DATABASE_URL is required for persistent conversation memory."
        )

    from langgraph.checkpoint.postgres import PostgresSaver

    with PostgresSaver.from_conn_string(database_url) as checkpointer:
        checkpointer.setup()
        application.state.checkpointer = checkpointer
        application.state.conversation_graph = build_graph(
            llm=application.state.llm,
            embedding_model=application.state.embedding_model,
            checkpointer=checkpointer,
        )
        yield


app = FastAPI(
    title="GitContext API",
    description=(
        "API for GitContext, a tool for analyzing Git repositories and "
        "extracting contextual information."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _set_status(repo_name: str, status: str, message: str, error: str | None = None) -> None:
    _INGESTION_STATUS[repo_name] = {
        "repo_name": repo_name,
        "status": status,
        "message": message,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _get_repo_list() -> list[str]:
    embedding_model = getattr(app.state, "embedding_model", None)
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model is not ready.")
    try:
        vector_store = SupabaseVectorStore(embedding_model)
        return vector_store.list_repositories()
    except ValueError as error:
        raise HTTPException(status_code=503, detail="Vector store is not configured.") from error


def _validate_repo_name(repo_name: str) -> str:
    repo_name = unquote(repo_name).strip()
    parts = repo_name.split("/")
    if len(parts) != 2 or not all(parts):
        raise HTTPException(
            status_code=400,
            detail="Repository identifier must use the format 'owner/repository'.",
        )
    return repo_name


@app.get("/")
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict[str, object]:
    model_ready = getattr(app.state, "embedding_model", None) is not None
    return {
        "status": "ok" if model_ready else "degraded",
        "embedding_model_ready": model_ready,
        "vector_store_configured": bool(os.getenv("SUPABASE_DATABASE_URL")),
    }


@app.post("/index")
def index(request: IngestRequest) -> dict[str, object]:
    try:
        repo_name = extract_github_repo(request.repo_url)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    _set_status(repo_name, "running", "Ingestion started.")

    try:
        result = ingest_repository(
            repo_url=request.repo_url,
            max_commits=request.max_commits,
            max_prs=request.max_prs,
            embedding_model=getattr(app.state, "embedding_model", None),
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
    repo_name = _validate_repo_name(repo_name)
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
    repo_name = _validate_repo_name(request.repo_name)

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
        graph = getattr(app.state, "conversation_graph", None)
        if graph is None:
            raise HTTPException(status_code=503, detail="Conversation graph is not ready.")
        thread_id = request.thread_id or str(uuid4())
        result = graph.invoke(
            {
                "thread_id": thread_id,
                "repo_name": repo_name,
                "question": request.question,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        if result.get("error") and result.get("answer") is None:
            raise RuntimeError("Conversation graph failed.")
        return {
            "thread_id": thread_id,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
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
    repo_name = _validate_repo_name(repo_name)
    try:
        embedding_model = getattr(app.state, "embedding_model", None)
        if embedding_model is None:
            raise HTTPException(status_code=503, detail="Embedding model is not ready.")
        vector_store = SupabaseVectorStore(embedding_model)
        deleted_count = vector_store.delete_repository(repo_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail="Repository deletion failed.") from error

    _INGESTION_STATUS.pop(repo_name, None)
    return {
        "repo_name": repo_name,
        "deleted_chunks": deleted_count,
        "status": "deleted",
    }
