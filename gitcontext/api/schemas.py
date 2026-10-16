from pydantic import BaseModel, Field,field_validator


class IngestRequest(BaseModel):
    repo_url: str  # https://github.com/owner/repo
    max_commits: int = Field(50, ge=0, le=5000)
    max_prs: int = Field(50, ge=0, le=200)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    repo_name: str = Field(min_length=1)
    thread_id: str | None = Field(default=None, min_length=1)
    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty.")
        return value

    @field_validator("repo_name")
    @classmethod
    def validate_repo_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Repository name cannot be empty.")
        return value