# from __future__ import annotations

# import json
# import os
# from pathlib import Path
# import re
# import sys
# import time
# from typing import Any, Optional
# from uuid import uuid4

# from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import logfire
# from pydantic import BaseModel, Field

# from gitcontext.eval.deepeval_model import GitContextEvalModel
# from gitcontext.generation.llm import create_llm

# load_dotenv()

# # Configure logfire for observability
# logfire.configure(
#     service_name="gitcontext-dataset-gen",
#     send_to_logfire="if-token-present",
# )

# TARGET_FILES = [
#     "gitcontext/api/main.py",
#     "gitcontext/api/guardrails.py",
#     "gitcontext/generation/llm.py",
#     "gitcontext/generation/classify_intent.py",
#     "gitcontext/generation/generation.py",
#     "gitcontext/graph/graph.py",
#     "gitcontext/graph/nodes.py",
#     "gitcontext/retrieval/hybrid_retriever.py",
#     "gitcontext/reranking/reranker.py",
#     "gitcontext/chunking/splitter.py",
#     "gitcontext/chunking/ast_chunker.py",
#     "README.md",
# ]


# class GoldenQAPair(BaseModel):
#     """Schema for fallback structured golden Q&A generation."""
#     input: str = Field(description="Realistic developer query about codebase implementation or architecture.")
#     expected_output: str = Field(description="Comprehensive and accurate ground-truth answer based on context.")


# class GoldenDatasetItem(BaseModel):
#     """Output schema for items stored in eval_golden_dataset.json."""
#     id: str
#     input: str
#     expected_output: str
#     context: list[str]
#     source_file: str


# def extract_repository_chunks(
#     repo_root: Path,
#     target_files: list[str] = TARGET_FILES,
#     chunk_size: int = 1200,
#     chunk_overlap: int = 150,
# ) -> list[dict[str, str]]:
#     """Scans repository files and extracts representative, self-contained context chunks."""
#     with logfire.span("Extract repository context chunks"):
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
#         )

#         selected_chunks: list[dict[str, str]] = []

#         for rel_path_str in target_files:
#             file_path = repo_root / rel_path_str
#             if not file_path.exists():
#                 logfire.warning("Target file not found, skipping: {path}", path=rel_path_str)
#                 continue

#             try:
#                 content = file_path.read_text(encoding="utf-8")
#             except Exception as e:
#                 logfire.error("Failed to read file {path}: {err}", path=rel_path_str, err=str(e))
#                 continue

#             if not content.strip() or len(content.strip()) < 50:
#                 continue

#             chunks = splitter.split_text(content)
#             if not chunks:
#                 continue

#             # Pick the most information-dense chunk (preferring code blocks with def/class)
#             best_chunk = chunks[0]
#             for candidate in chunks:
#                 if "def " in candidate or "class " in candidate or "## " in candidate:
#                     if len(candidate) > len(best_chunk) // 2:
#                         best_chunk = candidate
#                         break

#             formatted_chunk = f"File: {rel_path_str}\n\n{best_chunk.strip()}"
#             selected_chunks.append(
#                 {
#                     "source_file": rel_path_str,
#                     "chunk": formatted_chunk,
#                 }
#             )

#         logfire.info(
#             "Extracted {count} context chunks from repository",
#             count=len(selected_chunks),
#         )
#         return selected_chunks


# def generate_fallback_golden(chunk_text: str, source_file: str) -> Optional[GoldenQAPair]:
#     """Fallback generator using structured prompting via create_llm()."""
#     with logfire.span("Structured LLM fallback generation for {file}", file=source_file):
#         llm = create_llm(temperature=0.1)
#         system_prompt = (
#             "You are an expert AI test engineer for 'GitContext', a production RAG application for codebases.\n"
#             "Given the source code context below, generate ONE realistic developer question and ONE comprehensive,\n"
#             "technically exact ground-truth answer directly supported by the context.\n"
#             "The question must sound like an authentic software engineer querying codebase architecture.\n"
#             "You must respond ONLY with a valid JSON object matching this schema:\n"
#             "{\n"
#             '  "input": "<developer question>",\n'
#             '  "expected_output": "<ground-truth technical answer>"\n'
#             "}\n"
#             "Do NOT include markdown fences, greetings, or explanations outside the JSON."
#         )

#         user_prompt = f"Source File Context:\n{chunk_text}"

#         for attempt in range(1, 4):
#             try:
#                 resp = llm.invoke([
#                     SystemMessage(content=system_prompt),
#                     HumanMessage(content=user_prompt),
#                 ])
#                 raw_text = resp.content if isinstance(resp.content, str) else str(resp.content)
#                 cleaned = raw_text.strip()
#                 if "```json" in cleaned:
#                     cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
#                 elif "```" in cleaned:
#                     cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

#                 # Extract JSON object with regex if needed
#                 match = re.search(r"\{.*\}", cleaned, re.DOTALL)
#                 if match:
#                     cleaned = match.group(0)

#                 data = json.loads(cleaned)
#                 return GoldenQAPair.model_validate(data)
#             except Exception as e:
#                 logfire.warning(
#                     "Fallback attempt {attempt} failed for {file}: {err}",
#                     attempt=attempt,
#                     file=source_file,
#                     err=str(e),
#                 )
#                 time.sleep(2.0 * attempt)

#         return None


# def generate_dataset(
#     repo_root: Optional[Path] = None,
#     output_path: Optional[Path] = None,
# ) -> list[GoldenDatasetItem]:
#     """End-to-end dataset generation pipeline using DeepEval Synthesizer with automatic fallback."""
#     if repo_root is None:
#         repo_root = Path(__file__).resolve().parent.parent.parent
#     if output_path is None:
#         output_path = repo_root / "tests" / "eval_golden_dataset.json"

#     # Step 1: Extract clean context chunks
#     selected_chunks = extract_repository_chunks(repo_root)
#     if not selected_chunks:
#         raise RuntimeError("No context chunks could be extracted from repository.")

#     dataset_items: list[GoldenDatasetItem] = []

#     # Step 2: Attempt DeepEval Synthesizer generation
#     try:
#         from deepeval.synthesizer import Synthesizer
#         from deepeval.synthesizer.config import EvolutionConfig, StylingConfig

#         logfire.info("Initializing DeepEval Synthesizer with GitContextEvalModel")
#         eval_model = GitContextEvalModel()

#         styling_config = StylingConfig(
#             scenario="Software developers and DevOps engineers querying the GitContext codebase repository to inspect its architecture, retrieval strategies, router configuration, and guardrails.",
#             task="Generate technical developer queries and accurate, detailed ground-truth answers based on the code context.",
#             input_format="A natural developer question asking how a specific feature, class, or flow works in GitContext.",
#             expected_output_format="A technically accurate, thorough answer referencing the code implementation details.",
#         )
#         evolution_config = EvolutionConfig(num_evolutions=1)

#         synthesizer = Synthesizer(
#             model=eval_model,
#             async_mode=False,
#             styling_config=styling_config,
#             evolution_config=evolution_config,
#         )

#         with logfire.span("Execute DeepEval Synthesizer"):
#             goldens = synthesizer.generate_goldens_from_contexts(
#                 contexts=[[item["chunk"]] for item in selected_chunks],
#                 max_goldens_per_context=1,
#                 include_expected_output=True,
#                 source_files=[item["source_file"] for item in selected_chunks],
#             )

#         logfire.info("DeepEval Synthesizer generated {count} goldens", count=len(goldens))

#         for idx, golden in enumerate(goldens):
#             src_file = (
#                 golden.source_file
#                 if getattr(golden, "source_file", None)
#                 else (selected_chunks[idx]["source_file"] if idx < len(selected_chunks) else "unknown")
#             )
#             ctx = golden.context if golden.context else [selected_chunks[idx]["chunk"]]
#             dataset_items.append(
#                 GoldenDatasetItem(
#                     id=str(uuid4()),
#                     input=golden.input,
#                     expected_output=golden.expected_output or "",
#                     context=ctx,
#                     source_file=src_file,
#                 )
#             )

#     except Exception as e:
#         logfire.error("DeepEval Synthesizer encountered an issue: {err}. Switching to fallback.", err=str(e))
#         print(f"\n[Notice] DeepEval Synthesizer run hit an issue ({e}). Activating structured Pydantic fallback...")

#     # Step 3: Check if any chunks are missing or if Synthesizer produced fewer items than chunks
#     generated_sources = {item.source_file for item in dataset_items}
#     for item in selected_chunks:
#         if item["source_file"] not in generated_sources:
#             logfire.info("Generating fallback golden for chunk: {file}", file=item["source_file"])
#             qa = generate_fallback_golden(item["chunk"], item["source_file"])
#             if qa:
#                 dataset_items.append(
#                     GoldenDatasetItem(
#                         id=str(uuid4()),
#                         input=qa.input,
#                         expected_output=qa.expected_output,
#                         context=[item["chunk"]],
#                         source_file=item["source_file"],
#                     )
#                 )

#     # Step 4: Serialize into JSON
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     serialized = [item.model_dump() for item in dataset_items]

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(serialized, f, indent=2, ensure_ascii=False)

#     logfire.info("Successfully generated and saved {count} goldens to {path}", count=len(dataset_items), path=str(output_path))
#     return dataset_items


# def main() -> None:
#     # Ensure stdout handles UTF-8 characters on Windows consoles
#     if hasattr(sys.stdout, "reconfigure"):
#         sys.stdout.reconfigure(encoding="utf-8")

#     print("=" * 70)
#     print("GitContext - Synthetic Golden Dataset Generation Pipeline")
#     print("=" * 70)

#     start_time = time.time()
#     items = generate_dataset()
#     elapsed = time.time() - start_time

#     print(f"\n[Completed in {elapsed:.1f}s]")
#     print(f"Total Golden Q&A Pairs Generated: {len(items)}")
#     print(f"Output File: tests/eval_golden_dataset.json")

#     print("\n" + "=" * 70)
#     print("Sample Q&A Pairs from Generated Golden Dataset:")
#     print("=" * 70)

#     for i, item in enumerate(items[:3], start=1):
#         print(f"\n--- [Sample #{i} | Source: {item.source_file}] ---")
#         print(f"Q: {item.input}")
#         print(f"A: {item.expected_output}")

#     print("\n" + "=" * 70)


# if __name__ == "__main__":
#     main()


from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from uuid import uuid4

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, StylingConfig
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logfire
from pydantic import BaseModel, Field

from gitcontext.eval.deepeval_model import GitContextEvalModel
from gitcontext.generation.llm import create_llm

load_dotenv()

logfire.configure(
    service_name="gitcontext-dataset-gen",
    send_to_logfire="if-token-present",
)

# TARGET_FILES = [
#     "gitcontext/api/main.py",
#     "gitcontext/api/guardrails.py",
#     "gitcontext/generation/llm.py",
#     "gitcontext/generation/classify_intent.py",
#     "gitcontext/generation/generation.py",
#     "gitcontext/graph/graph.py",
#     "gitcontext/graph/nodes.py",
#     "gitcontext/retrieval/hybrid_retriever.py",
#     "gitcontext/reranking/reranker.py",
#     "gitcontext/chunking/splitter.py",
#     "gitcontext/chunking/ast_chunker.py",
#     "README.md",
# ]

TARGET_FILES = [
    "gitcontext/graph/states.py",
    "gitcontext/ingestion/commit_ingestion.py",
    "gitcontext/ingestion/discovery.py",
    "gitcontext/ingestion/pr_ingestion.py",
    "gitcontext/vectorstore/supabase_store.py",
]


class GoldenQAPair(BaseModel):
    input: str = Field(description="Realistic developer query about codebase implementation or architecture.")
    expected_output: str = Field(description="Comprehensive ground-truth answer strictly based on the context.")


class GoldenDatasetItem(BaseModel):
    id: str
    input: str
    expected_output: str
    context: list[str]
    source_file: str


FALLBACK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI test engineer for 'GitContext', a RAG app for Git repositories. "
        "Generate ONE realistic developer question and ONE accurate ground-truth answer from the code context.",
    ),
    ("human", "{chunk_text}"),
])


@logfire.instrument("Extract repository context chunks")
def extract_repository_chunks(repo_root: Path) -> list[dict[str, str]]:
    """Extracts one representative chunk per target file."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\nclass ", "\ndef ", "\n\n", "\n"],
    )

    selected_chunks: list[dict[str, str]] = []
    for rel_path in TARGET_FILES:
        file_path = repo_root / rel_path
        if not file_path.exists():
            continue

        content = file_path.read_text(encoding="utf-8").strip()
        chunks = splitter.split_text(content) if len(content) >= 50 else []
        if not chunks:
            continue

        # Prefer chunks containing class/function definitions or markdown headers
        best_chunk = next(
            (c for c in chunks if any(k in c for k in ("def ", "class ", "## "))),
            chunks[0],
        )
        selected_chunks.append({
            "source_file": rel_path,
            "chunk": f"File: {rel_path}\n\n{best_chunk.strip()}",
        })

    return selected_chunks


def generate_fallback_golden(chunk_text: str, source_file: str) -> Optional[GoldenQAPair]:
    """Fallback generator using LangChain's native with_structured_output."""
    try:
        chain = FALLBACK_PROMPT | create_llm(temperature=0.1).with_structured_output(GoldenQAPair)
        return chain.invoke({"chunk_text": chunk_text})
    except Exception as e:
        logfire.warning("Fallback failed for {file}: {err}", file=source_file, err=str(e))
        return None


@logfire.instrument("Generate Synthetic Golden Dataset")
def generate_dataset(
    repo_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> list[GoldenDatasetItem]:
    repo_root = repo_root or Path(__file__).resolve().parent.parent.parent
    output_path = output_path or (repo_root / "tests" / "eval_golden_dataset_1.json")

    selected_chunks = extract_repository_chunks(repo_root)
    dataset_items: list[GoldenDatasetItem] = []

    # 1. Primary path: DeepEval Synthesizer
    try:
        synthesizer = Synthesizer(
            model=GitContextEvalModel(),
            async_mode=False,
            styling_config=StylingConfig(
                scenario="Developers querying the GitContext repository about its architecture and implementation.",
                task="Generate developer questions and accurate ground-truth answers from code chunks.",
            ),
            evolution_config=EvolutionConfig(num_evolutions=1),
        )

        goldens = synthesizer.generate_goldens_from_contexts(
            contexts=[[item["chunk"]] for item in selected_chunks],
            max_goldens_per_context=1,
            include_expected_output=True,
            source_files=[item["source_file"] for item in selected_chunks],
        )

        for idx, g in enumerate(goldens):
            dataset_items.append(
                GoldenDatasetItem(
                    id=str(uuid4()),
                    input=g.input,
                    expected_output=g.expected_output or "",
                    context=g.context or [selected_chunks[idx]["chunk"]],
                    source_file=getattr(g, "source_file", None) or selected_chunks[idx]["source_file"],
                )
            )
    except Exception as e:
        logfire.warning("Synthesizer issue ({err}); filling remaining items with fallback.", err=str(e))

    # 2. Fallback path: Fill any missing files using structured output
    covered_files = {item.source_file for item in dataset_items}
    for item in selected_chunks:
        if item["source_file"] not in covered_files:
            if qa := generate_fallback_golden(item["chunk"], item["source_file"]):
                dataset_items.append(
                    GoldenDatasetItem(
                        id=str(uuid4()),
                        input=qa.input,
                        expected_output=qa.expected_output,
                        context=[item["chunk"]],
                        source_file=item["source_file"],
                    )
                )

    # 3. Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([item.model_dump() for item in dataset_items], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dataset_items


if __name__ == "__main__":
    items = generate_dataset()
    print(f"Generated {len(items)} golden Q&A pairs -> tests/eval_golden_dataset.json\n")
    for i, item in enumerate(items[:3], start=1):
        print(f"[{i}] ({item.source_file})\nQ: {item.input}\nA: {item.expected_output}\n")