from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import httpx
import logfire

logfire.configure(
    service_name="gitcontext-eval-collector",
    send_to_logfire="if-token-present",
)


def extract_answer_and_contexts(response_json: dict[str, Any]) -> tuple[str, list[str]]:
    """Extracts the LLM answer and retrieved contexts from the FastAPI JSON response."""
    # Check common response field names used in FastAPI RAG endpoints
    answer = (
        response_json.get("answer")
        or response_json.get("response")
        or response_json.get("output")
        or response_json.get("content")
        or ""
    )

    # Also capture retrieved chunks if your API returns them (useful for DeepEval Faithfulness)
    raw_contexts = (
        response_json.get("retrieval_context")
        or response_json.get("contexts")
        or response_json.get("sources")
        or response_json.get("chunks")
        or []
    )

    retrieved_contexts: list[str] = []
    if isinstance(raw_contexts, list):
        for item in raw_contexts:
            if isinstance(item, str):
                retrieved_contexts.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("content")
                    or item.get("page_content")
                    or item.get("text")
                    or item.get("chunk")
                    or json.dumps(item)
                )
                retrieved_contexts.append(str(text))

    return str(answer).strip(), retrieved_contexts


@logfire.instrument("Collect RAG Responses from FastAPI")
def collect_responses(
    dataset_path: Path,
    api_url: str = "http://localhost:8000/chat",
    repo_id: str | None = None,
    delay_seconds: float = 2.0,
    overwrite: bool = False,
) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    items: list[dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    total = len(items)
    print(f"Loaded {total} questions from {dataset_path}")
    print(f"Target FastAPI Endpoint: {api_url}\n" + "=" * 65)

    with httpx.Client(timeout=120.0) as client:
        for idx, item in enumerate(items, start=1):
            question = item.get("input", "")

            # Skip if already populated (unless --overwrite is passed)
            if not overwrite and item.get("actual_output"):
                print(f"[{idx}/{total}] Skipping (already has actual_output): {question[:60]}...")
                continue

            # Build request payload matching common FastAPI chat schemas
            payload: dict[str, Any] = {
                "question": question,
                "query": question,
                "message": question,
                "repo_name": repo_id or "harwinder766/gitcontext",
            }
            if repo_id:
                payload["repo_id"] = repo_id
                payload["repo_name"] = repo_id

            print(f"[{idx}/{total}] Asking: {question}")
            start_t = time.time()

            try:
                response = client.post(api_url, json=payload)
                response.raise_for_status()
                data = response.json()

                actual_output, retrieved_contexts = extract_answer_and_contexts(data)
                elapsed = round(time.time() - start_t, 2)

                item["actual_output"] = actual_output
                if retrieved_contexts:
                    item["retrieval_context"] = retrieved_contexts
                elif "context" in item and "retrieval_context" not in item:
                    # Fallback to synthetic context if API response doesn't expose chunks
                    item["retrieval_context"] = item["context"]

                item["latency_seconds"] = elapsed
                print(f"   -> Success ({elapsed}s) | Answer length: {len(actual_output)} chars")

            except httpx.HTTPStatusError as http_err:
                err_detail = http_err.response.text
                print(f"   -> HTTP {http_err.response.status_code} Error: {err_detail}")
                item["actual_output"] = ""
                item["error"] = f"HTTP {http_err.response.status_code}: {err_detail}"
            except Exception as err:
                print(f"   -> Request Failed: {err}")
                item["actual_output"] = ""
                item["error"] = str(err)

            # Save progress immediately after each question so nothing is lost on crash/rate-limit
            dataset_path.write_text(
                json.dumps(items, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            if idx < total:
                time.sleep(delay_seconds)

    print("=" * 65)
    print(f"Finished! Updated dataset saved to: {dataset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query local FastAPI backend and append answers to golden dataset.")
    parser.add_argument(
        "--file",
        type=str,
        default="tests/eval_golden_dataset.json",
        help="Path to the golden dataset JSON file (e.g. tests/eval_golden_dataset_1.json)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:8000/chat",
        help="Full URL of your FastAPI query/chat endpoint",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Optional repository name/ID if required by your endpoint payload",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between requests to avoid LLM rate limits",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite existing actual_output fields",
    )

    args = parser.parse_args()
    collect_responses(
        dataset_path=Path(args.file),
        api_url=args.url,
        repo_id=args.repo,
        delay_seconds=args.delay,
        overwrite=args.overwrite,
    )