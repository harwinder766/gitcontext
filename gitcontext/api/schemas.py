from pydantic import BaseModel, Field

class IngestRequest(BaseModel):
    repo_url: str                                  # https://github.com/owner/repo
    max_prs: int = Field(50, ge=0, le=200)

class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    repo_name: str