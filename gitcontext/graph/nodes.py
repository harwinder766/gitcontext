from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import logfire
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from gitcontext.generation.classify_intent import ClassifyIntent, PlannerDecision
from gitcontext.generation.generation import RAG, RAGResponse
from gitcontext.generation.normal_message import Answer, NormalMessage
from gitcontext.graph.states import ConversationState

MAX_HISTORY_MESSAGES = 6
MAX_ASSISTANT_HISTORY_CHARS = 1000


def _history(state: ConversationState) -> list[dict[str, Any]]:
    return list(state.get("messages", []))[-MAX_HISTORY_MESSAGES:]


def _append_turn(state: ConversationState, answer: str) -> list[dict[str, Any]]:
    messages = _history(state)
    # Trim assistant answer in history to prevent context window explosion
    trimmed_answer = answer
    if len(trimmed_answer) > MAX_ASSISTANT_HISTORY_CHARS:
        trimmed_answer = (
            trimmed_answer[:MAX_ASSISTANT_HISTORY_CHARS].rstrip()
            + "\n...[truncated for conversation context]"
        )

    messages.extend(
        [
            {"role": "user", "content": state.get("question", "")},
            {"role": "assistant", "content": trimmed_answer},
        ]
    )
    return messages[-MAX_HISTORY_MESSAGES:]


def _fallback_intent(question: str) -> PlannerDecision:
    normalized = re.sub(r"\s+", " ", question.strip().lower())
    if re.fullmatch(
        r"(hi|hello|hey|how are you(?: doing)?|good morning|good afternoon|good evening)[!.?]*",
        normalized,
    ):
        return PlannerDecision(intent="conversational", search_query=question)
    return PlannerDecision(intent="repository", search_query=question)


def classify_intent_node(
    state: ConversationState,
    llm: Any,
) -> dict[str, Any]:
    """Classify intent and rewrite query using conversation history."""
    question = state.get("question", "")
    history = _history(state)

    with logfire.span("🧠 Planner & Query Rewriter", question=question) as span:
        try:
            decision = ClassifyIntent(history, question, llm).classify()
            span.set_attributes({
                "intent": decision.intent,
                "search_query": decision.search_query,
                "used_fallback": False,
            })
            return {
                "intent": decision.intent,
                "current_query": decision.search_query,
                "error": None,
            }
        except Exception as error:
            decision = _fallback_intent(question)
            logfire.warn(
                "Classifier failed ({error}); using fallback intent '{intent}'",
                error=str(error),
                intent=decision.intent,
            )
            span.set_attributes({
                "intent": decision.intent,
                "search_query": decision.search_query,
                "used_fallback": True,
                "error": str(error),
            })
            return {
                "intent": decision.intent,
                "current_query": decision.search_query,
                "error": str(error),
            }

def greeting_node(
    state: ConversationState,
    llm: Any,
) -> dict[str, Any]:
    """Handle conversational messages using conversation history and NormalMessage."""
    question = state.get("question", "")
    history = _history(state)

    with logfire.span("✍️ Conversational Synthesis", question=question):
        try:
            normalizer = NormalMessage(
                query=question,
                messages=history,
                llm=llm,
            )
            response = normalizer.generate_answer()
            answer = response.answer
            error = None
        except Exception as error_value:
            answer = "I ran into an issue processing your message."
            error = str(error_value)
            logfire.error("Conversational node failed", error=error)

        return {
            "answer": answer,
            "sources": [],
            "messages": _append_turn(state, answer),
            "error": error,
        }
def repository_node(
    state: ConversationState,
    rag_factory: Callable[..., RAG],
    llm: Any,
    embedding_model: Any,
    cross_encoder_model: Any,
) -> dict[str, Any]:
    """Execute RAG retrieval using the rewritten search query."""
    original_question = state.get("question", "")
    search_query = state.get("current_query", original_question)
    repo_name = state.get("repo_name", "")
    history = _history(state)

    with logfire.span(
        "🔍 Repository RAG & Synthesis",
        original_question=original_question,
        search_query=search_query,
        repo_name=repo_name,
    ) as span:
        try:
            if not repo_name:
                raise ValueError("No repository selected (`repo_name` is empty).")

            response: RAGResponse = rag_factory(
                query=search_query,
                question=original_question,
                filter={"repo_name": repo_name},
                embedding_model=embedding_model,
                cross_encoder_model=cross_encoder_model,
                llm=llm,
                conversation_history=history,
            ).generate_response()

            sources = [source.model_dump() for source in response.sources]
            span.set_attribute("source_count", len(sources))

            return {
                "answer": response.answer,
                "sources": sources,
                "messages": _append_turn(state, response.answer),
                "error": None,
            }
        except Exception as error_value:
            answer = "I encountered an error while searching the repository."
            error = str(error_value)
            logfire.error(
                "Repository RAG node failed for repo '{repo_name}'",
                repo_name=repo_name,
                error=error,
            )
            span.set_attribute("error", error)
            return {
                "answer": answer,
                "sources": [],
                "messages": _append_turn(state, answer),
                "error": error,
            }