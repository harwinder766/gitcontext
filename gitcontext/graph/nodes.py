# from __future__ import annotations

# import re
# from collections.abc import Callable
# from typing import Any

# import logfire

# from gitcontext.generation.classify_intent import ClassifyIntent
# from gitcontext.generation.generation import RAG, RAGResponse
# from gitcontext.generation.normal_message import NormalMessage
# from gitcontext.graph.states import ConversationState

# MAX_HISTORY_MESSAGES = 6
# VALID_INTENTS = {
#     "greeting",
#     "history_current_repo",
#     "clarification",
# }


# def _history(state: ConversationState) -> list[dict[str, Any]]:
#     return list(state.get("messages", []))[-MAX_HISTORY_MESSAGES:]


# def _append_turn(state: ConversationState, answer: str) -> list[dict[str, Any]]:
#     messages = _history(state)
#     messages.extend(
#         [
#             {"role": "user", "content": state.get("question", "")},
#             {"role": "assistant", "content": answer},
#         ]
#     )
#     return messages[-MAX_HISTORY_MESSAGES:]


# def _fallback_intent(question: str, has_history: bool) -> str:
#     normalized = re.sub(r"\s+", " ", question.strip().lower())
#     if not normalized:
#         return "clarification"

#     if re.fullmatch(
#         r"(hi|hello|hey|how are you(?: doing)?|good morning|good afternoon|good evening)[!.?]*",
#         normalized,
#     ):
#         return "greeting"

#     # If the user refers to past conversation but there is no history, ask for clarification
#     if not has_history and any(
#         phrase in normalized
#         for phrase in (
#             "what did we discuss",
#             "what did i ask",
#             "what have we discussed",
#             "remember our conversation",
#             "that function",
#             "this function",
#             "explain that",
#         )
#     ):
#         return "clarification"

#     return "history_current_repo"


# def classify_intent_node(
#     state: ConversationState,
#     llm: Any,
# ) -> dict[str, Any]:
#     """Classify a turn, falling back safely when the classifier is unavailable."""
#     question = state.get("question", "")
#     has_history = bool(state.get("messages"))

#     with logfire.span(
#         "Node: classify_intent",
#         question=question,
#         has_history=has_history,
#     ) as span:
#         try:
#             intent = (
#                 ClassifyIntent(
#                     question,
#                     llm,
#                     has_history=has_history,
#                 )
#                 .classify()
#                 .intent
#             )
#             if intent not in VALID_INTENTS:
#                 raise ValueError(f"Unsupported intent: {intent}")

#             span.set_attributes({"intent": intent, "used_fallback": False})
#             return {"intent": intent, "error": None}

#         except Exception as error:
#             intent = _fallback_intent(question, has_history)
#             logfire.warn(
#                 "Classifier failed; using fallback intent '{intent}'",
#                 intent=intent,
#                 error=str(error),
#             )
#             span.set_attributes(
#                 {
#                     "intent": intent,
#                     "used_fallback": True,
#                     "error": str(error),
#                 }
#             )
#             return {"intent": intent, "error": str(error)}


# def greeting_node(
#     state: ConversationState,
#     llm: Any,
# ) -> dict[str, Any]:
#     question = state.get("question", "")

#     with logfire.span("Node: greeting", question=question) as span:
#         try:
#             answer = NormalMessage(
#                 query=question,
#                 llm=llm,
#             ).generate_answer().answer
#             error = None
#         except Exception as error_value:
#             answer = "I ran into an issue processing your message."
#             error = str(error_value)
#             logfire.error(
#                 "Greeting node failed",
#                 question=question,
#                 error=error,
#             )
#             span.set_attribute("error", error)

#         return {
#             "answer": answer,
#             "sources": [],
#             "messages": _append_turn(state, answer),
#             "error": error,
#         }


# def repository_node(
#     state: ConversationState,
#     rag_factory: Callable[..., RAG],
#     llm: Any,
#     embedding_model: Any,
#     cross_encoder_model: Any,
# ) -> dict[str, Any]:
#     question = state.get("question", "")
#     repo_name = state.get("repo_name", "")
#     history = _history(state)

#     with logfire.span(
#         "Node: repository",
#         question=question,
#         repo_name=repo_name,
#         history_length=len(history),
#     ) as span:
#         try:
#             if not repo_name:
#                 raise ValueError("No repository selected (`repo_name` is empty).")

#             response: RAGResponse = rag_factory(
#                 query=question,
#                 filter={"repo_name": repo_name},
#                 embedding_model=embedding_model,
#                 cross_encoder_model=cross_encoder_model,
#                 llm=llm,
#                 conversation_history=history,
#             ).generate_response()

#             sources = [source.model_dump() for source in response.sources]
#             span.set_attribute("source_count", len(sources))

#             return {
#                 "answer": response.answer,
#                 "sources": sources,
#                 "messages": _append_turn(state, response.answer),
#                 "error": None,
#             }
#         except Exception as error_value:
#             answer = "I encountered an error while searching the repository."
#             error = str(error_value)
#             logfire.error(
#                 "Repository RAG node failed for repo '{repo_name}'",
#                 repo_name=repo_name,
#                 question=question,
#                 error=error,
#             )
#             span.set_attribute("error", error)
#             return {
#                 "answer": answer,
#                 "sources": [],
#                 "messages": _append_turn(state, answer),
#                 "error": error,
#             }


# def clarification_node(state: ConversationState) -> dict[str, Any]:
#     with logfire.span(
#         "Node: clarification",
#         question=state.get("question", ""),
#     ):
#         answer = (
#             "Could you clarify what you are referring to? "
#             "You can ask about the current repository or something "
#             "we discussed earlier."
#         )
#         return {
#             "answer": answer,
#             "sources": [],
#             "messages": _append_turn(state, answer),
#             "error": None,
#         }

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import logfire
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from gitcontext.generation.classify_intent import ClassifyIntent, PlannerDecision
from gitcontext.generation.generation import RAG, RAGResponse
from gitcontext.generation.normal_message import NormalMessage
from gitcontext.graph.states import ConversationState

MAX_HISTORY_MESSAGES = 6


def _history(state: ConversationState) -> list[dict[str, Any]]:
    return list(state.get("messages", []))[-MAX_HISTORY_MESSAGES:]


def _append_turn(state: ConversationState, answer: str) -> list[dict[str, Any]]:
    messages = _history(state)
    messages.extend(
        [
            {"role": "user", "content": state.get("question", "")},
            {"role": "assistant", "content": answer},
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
    """Handle conversational messages using conversation history and native message objects."""
    question = state.get("question", "")
    history = _history(state)

    with logfire.span("✍️ Conversational Synthesis", question=question):
        try:
            # Build native LangChain message sequence
            system_prompt = (
                "You are the GitContext Assistant, a friendly and helpful conversational AI "
                "for GitContext, helping developers understand code repositories."
            )
            lc_messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
            for msg in history:
                if msg.get("role") == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))
            lc_messages.append(HumanMessage(content=question))

            response = llm.with_structured_output(NormalMessage).invoke(lc_messages)
            answer = getattr(response, "answer", str(response))
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