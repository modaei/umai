from __future__ import annotations

import json
from typing import Any

from umai.evals.models import EvaluationCase, MetricResult


METRIC_DEFAULT_THRESHOLDS = {
    "answer_relevancy": 0.7,
    "correctness": 0.7,
    "faithfulness": 0.7,
    "contextual_precision": 0.7,
    "contextual_recall": 0.7,
}


class DeepEvalUnavailableError(RuntimeError):
    """Raised when the optional DeepEval dependency is not installed."""


def run_deepeval_metrics(
    case: EvaluationCase,
    answer: str,
    tool_calls: list[dict[str, Any]],
) -> list[MetricResult]:
    """Run configured DeepEval metrics against one live UMAI answer."""

    if not case.metrics:
        return []
    try:
        from deepeval.metrics import (  # type: ignore
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
            GEval,
        )
        from deepeval.test_case import LLMTestCase, SingleTurnParams  # type: ignore
    except ImportError as exc:
        raise DeepEvalUnavailableError(
            "DeepEval is not installed. Install dev dependencies before running "
            "`umai-evaluate`."
        ) from exc

    retrieval_context = _retrieval_context(tool_calls)
    judge_context = _judge_context(tool_calls)
    expected_output = case.expected_output or "\n".join(case.required_phrases) or None
    test_case = LLMTestCase(
        input=case.question,
        actual_output=answer,
        retrieval_context=retrieval_context or None,
        context=judge_context or retrieval_context or None,
        expected_output=expected_output,
    )
    metric_factories = {
        "answer_relevancy": AnswerRelevancyMetric,
        "faithfulness": FaithfulnessMetric,
        "contextual_precision": ContextualPrecisionMetric,
        "contextual_recall": ContextualRecallMetric,
    }
    results: list[MetricResult] = []
    for metric_name in case.metrics:
        factory = metric_factories.get(metric_name)
        if factory is None and metric_name != "correctness":
            results.append(
                MetricResult(
                    name=metric_name,
                    score=None,
                    passed=False,
                    error=f"Unsupported DeepEval metric: {metric_name}",
                )
            )
            continue
        threshold = case.metric_thresholds.get(
            metric_name, METRIC_DEFAULT_THRESHOLDS[metric_name]
        )
        try:
            if metric_name == "correctness":
                if not expected_output:
                    results.append(
                        MetricResult(
                            name=metric_name,
                            score=None,
                            passed=False,
                            error="Correctness metric requires expected_output or required_phrases.",
                        )
                    )
                    continue
                metric = GEval(
                    name="Correctness",
                    threshold=threshold,
                    evaluation_params=[
                        SingleTurnParams.INPUT,
                        SingleTurnParams.ACTUAL_OUTPUT,
                        SingleTurnParams.EXPECTED_OUTPUT,
                        SingleTurnParams.CONTEXT,
                    ],
                    criteria=(
                        "Judge whether the actual UMAI response satisfies the expected "
                        "output or expected behavior for this operational query. Use the "
                        "provided context, including tool-call arguments and tool results, "
                        "when the visible answer is intentionally brief. Accept wording "
                        "differences, but fail if the answer omits required operational "
                        "facts, uses the wrong entity/date/scope, contradicts the expected "
                        "output, or only gives a generic non-answer."
                    ),
                )
            else:
                metric = factory(threshold=threshold)
            metric.measure(test_case)
            score = getattr(metric, "score", None)
            success = getattr(metric, "success", None)
            passed = bool(success) if success is not None else (
                score is not None and float(score) >= threshold
            )
            results.append(
                MetricResult(
                    name=metric_name,
                    score=float(score) if score is not None else None,
                    passed=passed,
                    reason=getattr(metric, "reason", None),
                )
            )
        except Exception as exc:
            results.append(
                MetricResult(
                    name=metric_name,
                    score=None,
                    passed=False,
                    error=str(exc),
                )
            )
    return results


def _judge_context(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Provide compact tool traces so correctness can judge terse UI answers."""

    context = []
    compact_calls = []
    for call in tool_calls:
        compact = {
            "tool": call.get("tool"),
            "arguments": call.get("arguments"),
        }
        result = call.get("result")
        if result is not None:
            compact["result"] = result
        compact_calls.append(compact)
    if compact_calls:
        context.append(f"Tool calls: {json.dumps(compact_calls, default=str)[:6000]}")
    context.extend(_retrieval_context(tool_calls))
    return context


def _retrieval_context(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Extract compact factual context from real tool traces for DeepEval."""

    context = []
    for call in tool_calls:
        result = call.get("result")
        if result is None:
            continue
        if isinstance(result, str):
            context.extend(_context_from_result_string(result))
        else:
            context.append(json.dumps(result, default=str)[:4000])
    return [item for item in context if item.strip()]


def _context_from_result_string(result: str) -> list[str]:
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return [result[:4000]]
    if isinstance(payload, dict):
        snippets = []
        for key in ("matches", "rows", "results", "well_tests", "readings"):
            value = payload.get(key)
            if value:
                snippets.append(json.dumps(value, default=str)[:4000])
        if snippets:
            return snippets
    return [json.dumps(payload, default=str)[:4000]]
