# from __future__ import annotations

# from typing import Literal, cast
# import logfire
# from langchain_openai import ChatOpenAI
# from pydantic import BaseModel, Field


# Intent = Literal[
#     "greeting",
#     "history_current_repo",
#     "clarification",
# ]


# class IntentClassification(BaseModel):
#     """Classification of a user's message for GitContext routing."""

#     intent: Intent = Field(
#         description=(
#             "The user's intent: 'greeting', 'history_current_repo', or 'clarification'."
#         )
#     )


# class ClassifyIntent:
#     """Classify a user's message to route it through GitContext."""

#     def __init__(
#         self,
#         query: str,
#         llm: ChatOpenAI,
#         has_history: bool = False,
#     ) -> None:
#         if not query or not query.strip():
#             raise ValueError("Query must be a non-empty string.")

#         if llm is None:
#             raise ValueError("LLM instance must be provided.")

#         self.query = query.strip()
#         self.llm = llm
#         self.has_history = has_history

#     def classify(self) -> IntentClassification:
#         """Return the structured intent classification."""
#         prompt = f"""You are an intent classifier for GitContext, a code repository question-answering application.

# Classify the user's message into exactly one of these categories:

# 1. greeting:
#    Greetings and ordinary social conversation that do not require repository retrieval or previous conversation history.

# 2. history_current_repo:
#    Questions that require repository retrieval, previous conversation context, or both.
#    Examples:
#    - "Explain the function we discussed earlier"
#    - "Compare this module with the one you mentioned before."
#    - "What does the function load_data() do in this repository?"

# 3. clarification:
#    Messages that cannot be routed confidently because the necessary reference or context is missing or genuinely ambiguous (e.g., referring to an earlier message when previous conversation is not available).

# Important rules:
# - A greeting followed by a repository question is NOT a greeting (route to history_current_repo).
# - If the user refers to previous conversation but `Previous conversation is available` is False, route to clarification.
# - Return only the structured classification.

# Previous conversation is available: {self.has_history}

# User message:
# {self.query}
# """

#         with logfire.span(
#             "Classify user intent",
#             query=self.query,
#             has_history=self.has_history,
#         ) as span:
#             structured_llm = self.llm.with_structured_output(IntentClassification)
#             try:
#                 result = structured_llm.invoke(prompt)
#                 classification = cast(IntentClassification, result)   
#                 # Attach the result directly to the span so it's filterable in Logfire
#                 span.set_attribute("intent", classification.intent)
#                 return classification
#             except Exception as error:
#                 raise RuntimeError(
#                     f"Intent classification failed: {error}"
#                 ) from error

from __future__ import annotations

from typing import Literal
import logfire
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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

    @logfire.instrument("Classify intent and rewrite query")
    def classify(self) -> PlannerDecision:
        # Build conversation history string for the prompt
        history_str = ""
        for msg in self.messages:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_str += f"{role}: {msg.get('content', '')}\n"

        system_prompt = (
            "You are an intelligent Assistant Planner and Query Rewriter for GitContext, "
            "a code repository question-answering application.\n\n"
            "Analyze the conversation history and the latest user message to make a routing decision."
        )

        user_prompt = f"""
CONVERSATION HISTORY:
{history_str if history_str else "No previous conversation."}

LATEST USER MESSAGE:
"{self.query}"

Task:
1. If the latest message is a greeting, pleasantry, or general question about GitContext itself that requires no repository code files, output intent as 'conversational'.
2. If it is a question about code, files, implementation details, commits, pull requests, or a follow-up referencing code discussed earlier, output intent as 'repository' AND rewrite the latest message into a standalone, fully-resolved search query.
"""

        lc_messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        structured_llm = self.llm.with_structured_output(PlannerDecision)
        try:
            result = structured_llm.invoke(lc_messages)
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