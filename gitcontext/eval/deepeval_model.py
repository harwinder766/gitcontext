# from __future__ import annotations

# import asyncio
# import json
# import os
# import random
# import re
# import time
# from typing import Any, Optional, Type

# from dotenv import load_dotenv
# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_core.messages import HumanMessage
# import logfire
# from pydantic import BaseModel

# from deepeval.models.base_model import DeepEvalBaseLLM
# from gitcontext.generation.llm import create_llm

# load_dotenv()


# class GitContextEvalModel(DeepEvalBaseLLM):
#     """DeepEval-compatible custom LLM wrapper for GitContext's LiteLLM Router.

#     Wraps the existing create_llm() pipeline to leverage its multi-provider
#     fallback routing (Groq, Gemini, xKiro) and local response caching, with added
#     exponential backoff for rate-limit resilience.
#     """

#     def __init__(
#         self,
#         model_name: str = "gitcontext-router-eval",
#         temperature: float = 0.0,
#         max_retries: int = 5,
#         backoff_factor: float = 2.0,
#         *args: Any,
#         **kwargs: Any,
#     ) -> None:
#         self._custom_model_name = model_name
#         self._temperature = temperature
#         self._max_retries = max_retries
#         self._backoff_factor = backoff_factor
#         super().__init__(model=model_name, *args, **kwargs)

#     def load_model(self, *args: Any, **kwargs: Any) -> BaseChatModel:
#         """Loads and returns GitContext's BaseChatModel via create_llm."""
#         # Ensure litellm uses local cost map without hanging on network fetching
#         os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
#         return create_llm(temperature=self._temperature)

#     def get_model_name(self, *args: Any, **kwargs: Any) -> str:
#         """Returns the registered evaluation model name."""
#         return self._custom_model_name or getattr(self, "name", "gitcontext-router-eval")

#     def (self, response_extract_text: Any) -> str:
#         """Safely extracts text string from BaseChatModel response."""
#         content = getattr(response, "content", response)
#         if isinstance(content, str):
#             return content.strip()
#         if isinstance(content, list):
#             parts: list[str] = []
#             for item in content:
#                 if isinstance(item, dict) and "text" in item:
#                     parts.append(item["text"])
#                 elif isinstance(item, str):
#                     parts.append(item)
#                 else:
#                     parts.append(str(item))
#             return "\n".join(parts).strip()
#         return str(content).strip()

#     def _parse_schema(self, text: str, schema: Type[BaseModel]) -> BaseModel:
#         """Attempts to parse structured output into the requested Pydantic schema."""
#         cleaned = text.strip()
#         # Strip markdown json fences if present
#         if "```json" in cleaned:
#             cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
#         elif "```" in cleaned:
#             cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

#         # 1. Direct model_validate_json
#         try:
#             return schema.model_validate_json(cleaned)
#         except Exception:
#             pass

#         # 2. Try json.loads with outermost bracket search
#         try:
#             match = re.search(r"\{.*\}", cleaned, re.DOTALL)
#             if match:
#                 data = json.loads(match.group(0))
#                 return schema.model_validate(data)
#         except Exception:
#             pass

#         # 3. Handle single-field wrapper schemas when model outputs plain text
#         fields = getattr(schema, "model_fields", {})
#         if len(fields) == 1:
#             field_name = next(iter(fields.keys()))
#             try:
#                 return schema(**{field_name: text.strip()})
#             except Exception:
#                 pass

#         if "response" in fields:
#             try:
#                 return schema(response=text.strip())
#             except Exception:
#                 pass

#         raise TypeError(
#             f"Model output could not be parsed into schema {schema.__name__}: {text[:120]}"
#         )

#     def generate(
#         self,
#         prompt: str,
#         schema: Optional[Type[BaseModel]] = None,
#         *args: Any,
#         **kwargs: Any,
#     ) -> Any:
#         """Synchronously generate LLM completion with exponential backoff retry.

#         If schema is provided, attempts to parse and return a Pydantic model instance.
#         If parsing fails with schema provided, raises TypeError so DeepEval's internal
#         trimAndLoadJson fallback handles it.
#         """
#         last_error: Optional[Exception] = None
#         for attempt in range(1, self._max_retries + 1):
#             try:
#                 msg = HumanMessage(content=prompt)
#                 resp = self.model.invoke([msg])
#                 text = self._extract_text(resp)

#                 if schema is not None:
#                     try:
#                         return self._parse_schema(text, schema)
#                     except Exception:
#                         # DeepEval checks for TypeError to trigger its own json loader
#                         raise TypeError(
#                             f"Model output could not be directly parsed into schema {schema.__name__}: {text[:120]}"
#                         )

#                 return text
#             except TypeError:
#                 # Re-raise schema parsing TypeError directly for DeepEval fallback
#                 raise
#             except Exception as e:
#                 last_error = e
#                 err_str = str(e).lower()
#                 is_rate_limit = any(
#                     term in err_str
#                     for term in [
#                         "429",
#                         "rate limit",
#                         "quota",
#                         "resource_exhausted",
#                         "timeout",
#                         "connection error",
#                         "internal server error",
#                     ]
#                 )
#                 if attempt < self._max_retries and is_rate_limit:
#                     sleep_time = (self._backoff_factor ** attempt) + random.uniform(0.5, 1.5)
#                     logfire.warning(
#                         "Transient error in GitContextEvalModel: {err}. Retrying in {s:.1f}s (attempt {a}/{m})",
#                         err=str(e),
#                         s=sleep_time,
#                         a=attempt,
#                         m=self._max_retries,
#                     )
#                     time.sleep(sleep_time)
#                 else:
#                     logfire.error("GitContextEvalModel generation failed: {err}", err=str(e))
#                     raise

#         if last_error:
#             raise last_error
#         return ""

#     async def a_generate(
#         self,
#         prompt: str,
#         schema: Optional[Type[BaseModel]] = None,
#         *args: Any,
#         **kwargs: Any,
#     ) -> Any:
#         """Asynchronously generate LLM completion with exponential backoff retry."""
#         last_error: Optional[Exception] = None
#         for attempt in range(1, self._max_retries + 1):
#             try:
#                 msg = HumanMessage(content=prompt)
#                 resp = await self.model.ainvoke([msg])
#                 text = self._extract_text(resp)

#                 if schema is not None:
#                     try:
#                         return self._parse_schema(text, schema)
#                     except Exception:
#                         raise TypeError(
#                             f"Model output could not be directly parsed into schema {schema.__name__}: {text[:120]}"
#                         )

#                 return text
#             except TypeError:
#                 raise
#             except Exception as e:
#                 last_error = e
#                 err_str = str(e).lower()
#                 is_rate_limit = any(
#                     term in err_str
#                     for term in [
#                         "429",
#                         "rate limit",
#                         "quota",
#                         "resource_exhausted",
#                         "timeout",
#                         "connection error",
#                         "internal server error",
#                     ]
#                 )
#                 if attempt < self._max_retries and is_rate_limit:
#                     sleep_time = (self._backoff_factor ** attempt) + random.uniform(0.5, 1.5)
#                     logfire.warning(
#                         "Transient error in GitContextEvalModel (async): {err}. Retrying in {s:.1f}s",
#                         err=str(e),
#                         s=sleep_time,
#                     )
#                     await asyncio.sleep(sleep_time)
#                 else:
#                     logfire.error("GitContextEvalModel async generation failed: {err}", err=str(e))
#                     raise

#         if last_error:
#             raise last_error
#         return ""

from __future__ import annotations

import os
from typing import Any, Optional, Type

from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from gitcontext.generation.llm import create_llm

load_dotenv()


class GitContextEvalModel(DeepEvalBaseLLM):
    """DeepEval wrapper around GitContext's LiteLLM Router."""

    def __init__(
        self,
        model_name: str = "gitcontext-router-eval",
        temperature: float = 0.0,
    ) -> None:
        self._temperature = temperature
        self.model_name = model_name
        super().__init__(model=model_name)

    def load_model(self) -> BaseChatModel:
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        return create_llm(temperature=self._temperature)

    def get_model_name(self) -> str:
        return self.model_name or "gitcontext-router-eval"

    def generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """Synchronously generate a response or structured Pydantic schema."""
        if schema is not None:
            try:
                structured_llm = self.model.with_structured_output(schema)
                return structured_llm.invoke(prompt)
            except Exception as e:
                # Raise TypeError so DeepEval's internal JSON parser fallback triggers
                raise TypeError(f"Structured output parsing failed: {e}") from e

        response = self.model.invoke(prompt)
        return str(response.content).strip()

    async def a_generate(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """Asynchronously generate a response or structured Pydantic schema."""
        if schema is not None:
            try:
                structured_llm = self.model.with_structured_output(schema)
                return await structured_llm.ainvoke(prompt)
            except Exception as e:
                raise TypeError(f"Async structured output parsing failed: {e}") from e

        response = await self.model.ainvoke(prompt)
        return str(response.content).strip()