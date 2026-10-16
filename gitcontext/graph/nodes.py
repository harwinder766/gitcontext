from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from gitcontext.generation.classify_intent import ClassifyIntent
from gitcontext.generation.generation import RAG, RAGResponse
from gitcontext.generation.normal_message import NormalMessage
from gitcontext.graph.states import ConversationState

MAX_HISTORY_MESSAGES = 6
VALID_INTENTS = {
    "greeting",
    # "history",
    # "current_repo",
    "history_current_repo",
    "clarification",
}


def _history(state: ConversationState) -> list[dict[str, Any]]:
    return list(state.get("messages", []))[-MAX_HISTORY_MESSAGES:]


def _append_turn(state: ConversationState, answer: str) -> list[dict[str, Any]]:
    messages = _history(state)
    messages.extend(
        [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer},
        ]
    )
    return messages[-MAX_HISTORY_MESSAGES:]


def _fallback_intent(question: str, has_history: bool) -> str:
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    if re.fullmatch(
        r"(hi|hello|hey|how are you(?: doing)?|good morning|good afternoon|good evening)[!.?]*",
        normalized,
    ):
        return "greeting"
    if any(
        phrase in normalized
        for phrase in (
            "what did we discuss",
            "what did i ask",
            "what have we discussed",
            "remember our conversation",
        )
    ):
        return "history"
    if has_history and any(
        phrase in normalized
        for phrase in (
            "that function",
            "this function",
            "explain that",
            "tell me more",
            "what about it",
            "which one",
        )
    ):
        return "history_current_repo"
    if normalized.startswith(("hi, ", "hello, ", "hey, ")):
        return "history_current_repo"
    if normalized:
        return "current_repo"
    return "clarification"


def classify_intent_node(
    state: ConversationState,
    llm: Any,
) -> dict[str, Any]:
    """Classify a turn, falling back safely when the classifier is unavailable."""
    try:
        intent = ClassifyIntent(
            state["question"],
            llm,
            has_history=bool(state.get("messages")),
        ).classify().intent
        if intent not in VALID_INTENTS:
            raise ValueError(f"Unsupported intent: {intent}")
    except Exception as error:
        intent = _fallback_intent(
            state["question"],
            bool(state.get("messages")),
        )
        return {"intent": intent, "error": str(error)}
    return {"intent": intent, "error": None}


def greeting_node(
    state: ConversationState,
    llm: Any,
) -> dict[str, Any]:
    try:
        answer = NormalMessage(
            query=state["question"],
            llm=llm,
        ).generate_answer().answer
        error = None
    except Exception as error_value:
        answer = "I ran into an issue processing your message."
        error = str(error_value)
    return {
        "answer": answer,
        "sources": [],
        "messages": _append_turn(state, answer),
        "error": error,
    }


# def history_node(state: ConversationState) -> dict[str, Any]:
#     questions = [
#         message["content"]
#         for message in _history(state)
#         if message.get("role") == "user"
#     ]
#     if not questions:
#         answer = "We have not discussed anything earlier in this conversation."
#     else:
#         answer = "Earlier in this conversation, you asked:\n" + "\n".join(
#             f"- {question}" for question in questions
#         )
#     return {
#         "answer": answer,
#         "sources": [],
#         "messages": _append_turn(state, answer),
#         "error": None,
#     }


def repository_node(
    state: ConversationState,
    rag_factory: Callable[..., RAG],
    llm: Any,
    embedding_model: Any,
) -> dict[str, Any]:
    response: RAGResponse = rag_factory(
        query=state["question"],
        filter={"repo_name": state["repo_name"]},
        embedding_model=embedding_model,
        llm=llm,
        conversation_history=_history(state),
    ).generate_response()
    sources = [source.model_dump() for source in response.sources]
    return {
        "answer": response.answer,
        "sources": sources,
        "messages": _append_turn(state, response.answer),
        "error": None,
    }


def clarification_node(state: ConversationState) -> dict[str, Any]:
    answer = (
        "Could you clarify what you are referring to? "
        "You can ask about the current repository or something "
        "we discussed earlier."
    )
    return {
        "answer": answer,
        "sources": [],
        "messages": _append_turn(state, answer),
        "error": None,
    }
