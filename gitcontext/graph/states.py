from __future__ import annotations

from typing import Any, TypedDict


class ConversationState(TypedDict, total=False):
    """Serializable state carried between graph nodes and checkpoints."""

    thread_id: str
    repo_name: str
    question: str
    messages: list[dict[str, Any]]
    intent: str
    retrieved_documents: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    error: str | None
