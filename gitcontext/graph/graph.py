from __future__ import annotations

from collections.abc import Callable
from typing import Any

import logfire
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from gitcontext.generation.generation import RAG
from gitcontext.graph.nodes import (
    classify_intent_node,
    greeting_node,
    repository_node,
)
from gitcontext.graph.states import ConversationState


def _route(state: ConversationState) -> str:
    intent = state.get("intent")
    if intent in ("conversational", "greeting"):
        route = "greeting"
    elif intent in ("repository", "history_current_repo"):
        route = "repository"
    else:
        route = "greeting"

    logfire.debug("Routing graph", intent=intent, target_node=route)
    return route


@logfire.instrument("Build LangGraph workflow")
def build_graph(
    *,
    llm: Any,
    embedding_model: Any = None,
    cross_encoder_model: Any | None = None,
    retriever: Any | None = None,
    rag_factory: Callable[..., RAG] = RAG,
    checkpointer: Any | None = None,
) -> CompiledStateGraph:
    """Compile the conversation graph with an injected checkpointer."""
    graph = StateGraph(ConversationState)
    graph.add_node(
        "classify",
        lambda state: classify_intent_node(state, llm),
    )
    graph.add_node(
        "greeting",
        lambda state: greeting_node(state, llm),
    )
    graph.add_node(
        "repository",
        lambda state: repository_node(
            state,
            rag_factory=rag_factory,
            llm=llm,
            embedding_model=embedding_model,
            cross_encoder_model=cross_encoder_model,
            retriever=retriever,
        ),
    )

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        _route,
        {
            "greeting": "greeting",
            "repository": "repository",
        },
    )
    graph.add_edge("greeting", END)
    graph.add_edge("repository", END)

    return graph.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver()
    )