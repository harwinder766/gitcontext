from __future__ import annotations

from typing import Literal, cast
import logfire
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class PlannerDecision(BaseModel):
    """Router and query rewriter for GitContext."""

    intent: Literal["conversational", "repository"] = Field(
        description=(
            "'conversational' for greetings, pleasantries, or general questions about "
            "what GitContext is. 'repository' for any question about code, files, "
            "commits, PRs, or follow-ups on previous technical answers."
        )
    )
    search_query: str = Field(
        description=(
            "If intent is 'repository', rewrite the user's message into a standalone, "
            "keyword-rich search query resolving any pronouns (it, that, this function) "
            "using the conversation history. If 'conversational', simply repeat the user's latest message."
        )
    )


def to_langchain_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert raw history dictionaries into LangChain BaseMessage instances."""
    lc_messages: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
    return lc_messages


CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an intelligent Assistant Planner and Query Rewriter for GitContext, "
                "a code repository question-answering application.\n\n"
                "Analyze the conversation history and the latest user message to make a routing decision.\n\n"
                "Task instructions:\n"
                "1. If the latest message is a greeting, pleasantry, or general question about GitContext itself that requires no repository code files, output intent as 'conversational'.\n"
                "2. If it is a question about code, files, implementation details, commits, pull requests, or a follow-up referencing code discussed earlier, output intent as 'repository' AND rewrite the latest message into a standalone, fully-resolved search query with essential technical keywords."
            ),
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ]
)


class ClassifyIntent:
    """Classify intent and rewrite queries for GitContext."""

    def __init__(
        self,
        messages: list[dict[str, str]],
        query: str,
        llm: ChatOpenAI,
    ) -> None:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if llm is None:
            raise ValueError("LLM instance must be provided.")

        self.messages = messages
        self.query = query.strip()
        self.llm = llm

    def classify(self) -> PlannerDecision:
        question = self.query
        with logfire.span("Classify intent and rewrite query",query = question) as span:
            structured_llm = self.llm.with_structured_output(PlannerDecision)
            chain = CLASSIFY_PROMPT | structured_llm
            try:
                result = chain.invoke(
                    {
                        "history": to_langchain_messages(self.messages),
                        "query":  question,
                    }
                )
                if not isinstance(result, PlannerDecision):
                    raise TypeError("Expected PlannerDecision from structured output.")
                logfire.info(
                    "Intent: {intent} | Rewritten Query: {query}",
                    intent=result.intent,
                    query=result.search_query,
                )
                return result
            except Exception as error:
                raise RuntimeError(f"Intent classification failed: {error}") from error