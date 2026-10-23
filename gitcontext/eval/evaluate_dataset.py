from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
import logfire

from gitcontext.eval.deepeval_model import GitContextEvalModel

logfire.configure(
    service_name="gitcontext-rag-evaluator",
    send_to_logfire="if-token-present",
)


def build_metrics(eval_model: GitContextEvalModel, threshold: float = 0.7) -> dict[str, Any]:
    """Initializes standard RAG Triad + Retrieval metrics using GitContextEvalModel."""
    # async_mode=False ensures metrics make sequential LLM calls to avoid 429 rate limits
    return {
        "faithfulness": FaithfulnessMetric(
            threshold=threshold,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=threshold,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        ),
        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=0.5,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        ),
        "contextual_precision": ContextualPrecisionMetric(
            threshold=threshold,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        ),
        "contextual_recall": ContextualRecallMetric(
            threshold=threshold,
            model=eval_model,
            include_reason=True,
            async_mode=False,
        ),
    }


@logfire.instrument("Run DeepEval RAG Evaluation")
def run_evaluation(
    dataset_path: Path,
    output_report_path: Path,
    selected_metrics: list[str],
    threshold: float = 0.7,
    delay_seconds: float = 2.0,
    overwrite: bool = False,
) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    items: list[dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
    eval_model = GitContextEvalModel()
    all_metrics = build_metrics(eval_model, threshold=threshold)

    active_metrics = {k: v for k, v in all_metrics.items() if k in selected_metrics}
    if not active_metrics:
        raise ValueError(f"No valid metrics selected. Choose from: {list(all_metrics.keys())}")

    print("=" * 75)
    print(f"Loaded {len(items)} items from: {dataset_path}")
    print(f"Active Metrics: {', '.join(active_metrics.keys())} (threshold={threshold})")
    print("=" * 75)

    try:
        for idx, item in enumerate(items, start=1):
            question = item.get("input", "")
            actual_output = item.get("actual_output", "")
            expected_output = item.get("expected_output", "")
            retrieval_context = item.get("retrieval_context") or item.get("context") or []

            if isinstance(retrieval_context, str):
                retrieval_context = [retrieval_context]

            if not actual_output:
                print(f"[{idx}/{len(items)}] Skipping (missing actual_output): {question[:60]}...")
                continue

            if not overwrite and "eval_scores" in item and all(m in item["eval_scores"] for m in active_metrics):
                print(f"[{idx}/{len(items)}] Skipping (already evaluated): {question[:60]}...")
                continue

            print(f"\n[{idx}/{len(items)}] Evaluating: {question[:70]}...")
            test_case = LLMTestCase(
                input=question,
                actual_output=actual_output,
                expected_output=expected_output,
                retrieval_context=retrieval_context,
            )

            item.setdefault("eval_scores", {})

            for metric_name, metric_obj in active_metrics.items():
                if not overwrite and metric_name in item["eval_scores"]:
                    continue

                start_t = time.time()
                try:
                    metric_obj.measure(test_case)
                    score = round(float(metric_obj.score or 0.0), 4)
                    passed = bool(score >= threshold)
                    reason = getattr(metric_obj, "reason", "") or ""
                    elapsed = round(time.time() - start_t, 2)

                    item["eval_scores"][metric_name] = {
                        "score": score,
                        "passed": passed,
                        "reason": reason,
                    }
                    status_str = "PASS" if passed else "FAIL"
                    print(f"   -> {metric_name:<22}: {score:.2f} [{status_str}] ({elapsed}s)")
                except Exception as e:
                    logfire.error("Metric {m} failed on item {i}: {err}", m=metric_name, i=idx, err=str(e))
                    print(f"   -> {metric_name:<22}: ERROR ({e})")
                    item["eval_scores"][metric_name] = {
                        "score": None,
                        "passed": False,
                        "error": str(e),
                    }

                time.sleep(delay_seconds)

            # Save intermediate results to the report file after each item
            _save_report(items, active_metrics, threshold, output_report_path)

    finally:
        summary = _save_report(items, active_metrics, threshold, output_report_path)
        _print_summary(summary, output_report_path)

def _save_report(
    items: list[dict[str, Any]],
    active_metrics: dict[str, Any],
    threshold: float,
    output_report_path: Path,
) -> dict[str, Any]:
    """Calculates aggregate scores across all items and writes JSON report."""
    summary_metrics: dict[str, Any] = {}

    for metric in active_metrics:
        # Collect all valid scores for this metric in a single pass
        scores = [
            item["eval_scores"][metric]["score"]
            for item in items
            if item.get("eval_scores", {}).get(metric, {}).get("score") is not None
        ]
        count = len(scores)
        passed = sum(1 for s in scores if s >= threshold)

        summary_metrics[metric] = {
            "average_score": round(sum(scores) / count, 4) if count else 0.0,
            "pass_rate": round((passed / count) * 100, 2) if count else 0.0,
            "evaluated_count": count,
            "threshold": threshold,
        }

    report_payload = {
        "summary": summary_metrics,
        "total_items": len(items),
        "results": items,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_metrics


def _print_summary(summary: dict[str, Any], output_report_path: Path) -> None:
    print("\n--- FINAL RAG EVALUATION SUMMARY ---")
    for metric, stats in summary.items():
        print(
            f"{metric:<22} | Avg: {stats['average_score']:.2f} | "
            f"Pass: {stats['pass_rate']:5.1f}% ({stats['evaluated_count']} items)"
        )
    print(f"\nReport saved to: {output_report_path}")


if __name__ == "__main__":
    # Direct configuration (No argparse needed)
    DATASET_FILE = Path("tests/eval_golden_dataset_combined_all_2_try.json")
    OUTPUT_REPORT = Path("tests/eval_results_report_1.json")
    METRICS = [
        "faithfulness",
        "answer_relevancy",
        "contextual_relevancy",
        "contextual_precision",
        "contextual_recall",
    ]

    run_evaluation(
        dataset_path=DATASET_FILE,
        output_report_path=OUTPUT_REPORT,
        selected_metrics=METRICS,
        threshold=0.7,
        delay_seconds=2.0,
        overwrite=False,
    )