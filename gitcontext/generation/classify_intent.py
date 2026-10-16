from __future__ import annotations
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


Intent = Literal[
    "greeting",
    # "history",
    # "current_repo",
    "history_current_repo",
    "clarification",
]


class IntentClassification(BaseModel):
    """Classification of a user's message for GitContext routing."""

    intent: Intent = Field(
        description=(
            "The user's intent: greeting,history_current_repo, or clarification."
        )
    )


class ClassifyIntent:
    """Classify a user's message to route it through GitContext."""

    def __init__(
        self,
        query: str,
        llm: ChatOpenAI,
        has_history: bool = False,
    ) -> None:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if llm is None:
            raise ValueError("LLM instance must be provided.")

        self.query = query.strip()
        self.llm = llm
        self.has_history = has_history

    def classify(self) -> IntentClassification:
        """Return the structured intent classification."""

        prompt = f"""
You are an intent classifier for GitContext, a code repository
question-answering application.

Classify the user's message into exactly one of these categories:

1. greeting:
   Greetings and ordinary social conversation that do not require
   repository retrieval or previous conversation history.

2. history_current_repo:
   Questions that require only repository retrieval or both previous conversation context and repository
   retrieval.
   Examples: "Explain the function we discussed earlier" or
   "Compare this module with the one you mentioned before." or
   "What does the function load_data() do in this repository?"

3. clarification:
   Messages that cannot be routed confidently because the necessary
   reference or context is missing or genuinely ambiguous.

Important rules:
- A greeting followed by a repository question is not a greeting.
- Use history_current_repo when both conversation context and
  repository retrieval are necessary.
- Do not assume a repository has been inspected merely because the
  user mentions a file or function.
- Return only the structured classification.

Previous conversation is available: {self.has_history}

User message:
{self.query}
"""

        structured_llm = self.llm.with_structured_output(
            IntentClassification
        )

        return structured_llm.invoke(prompt)
