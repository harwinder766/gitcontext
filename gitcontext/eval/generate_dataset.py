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