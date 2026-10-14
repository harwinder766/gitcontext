from fastapi import FastAPI, HTTPException
from gitcontext.ingestion.pipeline import ingest_repository
from gitcontext.api.schemas import IngestRequest, AskRequest
from gitcontext.generation.generation import RAG

import logging

app = FastAPI(
    title='GitContext API',
    description='API for GitContext, a tool for analyzing Git repositories and extracting contextual information.',
    version='1.0.0',
)

@app.get("/")
def status():
    return {"status": "ok"}

@app.post("/ingest")
def ingest(request: IngestRequest):
    """Ingest a GitHub repository."""
    try:
        return ingest_repository(
            repo_url=request.repo_url,
            max_commits=request.max_commits,
            max_prs=request.max_prs,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        # Log the real exception on the server in production.
        raise HTTPException(
            status_code=500,
            detail="Repository ingestion failed.",
        ) from error


# logger = logging.getLogger(__name__)

# @app.post("/ingest")
# def ingest(request: IngestRequest):
#     try:
#         return ingest_repository(
#             repo_url=request.repo_url,
#             max_prs=request.max_prs,
#         )
#     except ValueError as error:
#         raise HTTPException(
#             status_code=400,
#             detail=str(error),
#         ) from error
#     except Exception as error:
#         logger.exception("Repository ingestion failed")
#         raise HTTPException(
#             status_code=500,
#             detail="Repository ingestion failed. Check server logs.",
#         ) from error

@app.post("/query")
def query(request: AskRequest):
    """Query a GitHub repository."""
    try:
        rag = RAG(query = request.question, filter={"repo_name": request.repo_name},)
        return rag.generate_response()
        
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Query failed.",
        ) from error