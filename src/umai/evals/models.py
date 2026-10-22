from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationCase:
    """One live UMAI question and the checks expected for its answer."""

    id: str
    category: str
    question: str
    site_id: int
    current_date: str | None = None
    response_mode: str = "fast"
    deterministic: bool = False
    expected_output: str | None = None
    expected_tool_calls: tuple[dict[str, Any], ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ("answer_relevancy",)
    metric_thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeterministicCheckResult:
    """Result of a non-LLM assertion that is treated as a hard gate."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class MetricResult:
    """Compact DeepEval metric result stored in the local report file."""

    name: str
    score: float | None
    passed: bool
    reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Complete result for one live evaluation case."""

    case: EvaluationCase
    answer: str
    tool_calls: list[dict[str, Any]]
    stats: dict[str, Any]
    checks: list[DeterministicCheckResult]
    metrics: list[MetricResult]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks) and all(
            metric.passed for metric in self.metrics
        )
