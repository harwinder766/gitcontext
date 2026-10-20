# from __future__ import annotations

# from collections.abc import Callable
# from typing import Any

# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import END, START, StateGraph

# from gitcontext.generation.generation import RAG
# from gitcontext.graph.nodes import (
#     clarification_node,
#     classify_intent_node,
#     greeting_node,
#     repository_node,
# )
# from gitcontext.graph.states import ConversationState


# def _route(state: ConversationState) -> str:
#     intent = state.get("intent")
#     if intent == "greeting":
#         return "greeting"
#     # if intent == "history":
#     #     return "history"
#     elif intent == "history_current_repo":
#         return "repository"
#     return "clarification"


# def build_graph(
#     *,
#     llm: Any,
#     embedding_model: Any = None,
#     rag_factory: Callable[..., RAG] = RAG,
#     checkpointer: Any | None = None,
# ):
#     """Compile the conversation graph with an injected checkpointer."""
#     graph = StateGraph(ConversationState)
#     graph.add_node(
#         "classify",
#         lambda state: classify_intent_node(state, llm),
#     )
#     graph.add_node(
#         "greeting",
#         lambda state: greeting_node(state, llm),
#     )
#     # graph.add_node("history", history_node)
#     graph.add_node(
#         "repository",
#         lambda state: repository_node(
#             state,
#             rag_factory=rag_factory,
#             llm=llm,
#             embedding_model=embedding_model,

#         ),
#     )
#     graph.add_node("clarification", clarification_node)

#     graph.add_edge(START, "classify")
#     graph.add_conditional_edges(
#         "classify",
#         _route,
#         {
#             "greeting": "greeting",
#             # "history": "history",
#             "repository": "repository",
#             "clarification": "clarification",
#         },
#     )
#     for node in ("greeting", "repository", "clarification"):
#         graph.add_edge(node, END)

#     return graph.compile(checkpointer=checkpointer or MemorySaver())


from __future__ import annotations

from collections.abc import Callable
from typing import Any

import logfire
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sentence_transformers import CrossEncoder

from gitcontext.generation.generation import RAG
from gitcontext.graph.nodes import (
    clarification_node,
    classify_intent_node,
    greeting_node,
    repository_node,
)
from gitcontext.graph.states import ConversationState


def _route(state: ConversationState) -> str:
    intent = state.get("intent")
    if intent == "greeting":
        route = "greeting"
    elif intent == "history_current_repo":
        route = "repository"
    else:
        route = "clarification"

    logfire.debug("Routing graph", intent=intent, target_node=route)
    return route


@logfire.instrument("Build LangGraph workflow")
def build_graph(
    *,
    llm: Any,
    embedding_model: Any = None,
    cross_encoder_model: CrossEncoder | None = None,
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
        ),
    )
    graph.add_node("clarification", clarification_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        _route,
        {
            "greeting": "greeting",
            "repository": "repository",
            "clarification": "clarification",
        },
    )
    for node in ("greeting", "repository", "clarification"):
        graph.add_edge(node, END)

    return graph.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver()
    )