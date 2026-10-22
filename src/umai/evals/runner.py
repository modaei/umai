from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from umai.config.logging import configure_logging
from umai.config.settings import Settings
from umai.evals.cases import filter_cases, load_cases
from umai.evals.checks import run_deterministic_checks
from umai.evals.deepeval_runner import DeepEvalUnavailableError, run_deepeval_metrics
from umai.evals.models import EvaluationCase, EvaluationResult
from umai.services.chat_service import answer_chat


DEFAULT_OUTPUT_DIR = Path("eval-results")


def main(argv: list[str] | None = None) -> int:
    """Run live UMAI evaluations from a developer workstation."""

    args = _parse_args(argv)
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    try:
        _preflight(settings)
        _configure_deepeval_environment(settings)
        cases = filter_cases(
            load_cases(args.cases),
            category=args.category,
            case_id=args.case_id,
            limit=args.limit,
        )
        if not cases:
            raise RuntimeError("No evaluation cases matched the selected filters.")
        results = [
            _evaluate_case(settings, case, args.current_date)
            for case in cases
        ]
        report_path = _write_report(
            results,
            Path(args.output_dir),
            failed_only=args.failed_only,
        )
    except DeepEvalUnavailableError as exc:
        print(f"Evaluation setup failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 1

    passed = sum(1 for result in results if result.passed)
    total = len(results)
    print(f"UMAI evaluation complete: {passed}/{total} passed")
    print(f"Report: {report_path}")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.case.id}: {result.case.question}")
        for check in result.checks:
            if not check.passed:
                print(f"  check: {check.message}")
        for metric in result.metrics:
            if not metric.passed:
                detail = metric.error or metric.reason or "metric failed"
                print(f"  metric {metric.name}: {detail}")
    return 0 if passed == total else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local live DeepEval checks against UMAI."
    )
    parser.add_argument("--cases", default=None, help="Path to evaluation case JSON.")
    parser.add_argument("--category", default=None, help="Run only one case category.")
    parser.add_argument("--case-id", default=None, help="Run only one case id.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases.")
    parser.add_argument(
        "--current-date",
        default=None,
        help="Override current date for cases that do not define one.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for local JSON evaluation reports.",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Write only failed cases to the JSON report results array.",
    )
    return parser.parse_args(argv)


def _preflight(settings: Settings) -> None:
    """Fail fast when local live dependencies are not reachable."""

    settings.validate()
    settings.validate_database()
    _check_deepeval()
    _check_mysql(settings)
    _check_vector_db(settings)
    _check_omreports(settings)


def _check_deepeval() -> None:
    if importlib.util.find_spec("deepeval") is None:
        raise DeepEvalUnavailableError(
            "DeepEval is not installed. Install dev dependencies before running "
            "`umai-evaluate`."
        )


def _check_mysql(settings: Settings) -> None:
    url = URL.create(
        drivername="mysql+pymysql",
        username=settings.db_user,
        password=settings.db_password,
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
    )
    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def _check_vector_db(settings: Settings) -> None:
    url = URL.create(
        drivername="postgresql+psycopg",
        username=settings.vector_db_user,
        password=settings.vector_db_password,
        host=settings.vector_db_host,
        port=settings.vector_db_port,
        database=settings.vector_db_name,
    )
    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as connection:
        exists = connection.execute(
            text("SELECT to_regclass('public.rag_chunks') IS NOT NULL")
        ).scalar_one()
    if not exists:
        raise RuntimeError("Vector DB table public.rag_chunks does not exist.")


def _check_omreports(settings: Settings) -> None:
    # Omreports does not expose a guaranteed health endpoint in this package.
    # A lightweight GET is enough to catch connection refusal in local setup.
    try:
        with httpx.Client(timeout=3) as client:
            client.get(settings.omreports_api_url)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Omreports is not reachable: {exc}") from exc


def _configure_deepeval_environment(settings: Settings) -> None:
    """Expose UMAI's provider-neutral LLM config to OpenAI-compatible judges."""

    os.environ.setdefault("OPENAI_API_KEY", settings.llm_api_key)
    os.environ.setdefault("OPENAI_BASE_URL", settings.llm_base_url)
    os.environ.setdefault("OPENAI_API_BASE", settings.llm_base_url)


def _evaluate_case(
    settings: Settings,
    case: EvaluationCase,
    current_date_override: str | None,
) -> EvaluationResult:
    answer, tool_calls, stats = answer_chat(
        settings=settings,
        site_id=case.site_id,
        site_name=None,
        history=[],
        question=case.question,
        response_mode=case.response_mode,
        current_date=case.current_date or current_date_override,
    )
    checks = run_deterministic_checks(case, answer, tool_calls)
    metrics = [] if case.deterministic else run_deepeval_metrics(case, answer, tool_calls)
    return EvaluationResult(
        case=case,
        answer=answer,
        tool_calls=tool_calls,
        stats=stats,
        checks=checks,
        metrics=metrics,
    )


def _write_report(
    results: list[EvaluationResult],
    output_dir: Path,
    *,
    failed_only: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"umai-eval-{timestamp}.json"
    path.write_text(
        json.dumps(
            _report_payload(results, failed_only=failed_only),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


def _report_payload(
    results: list[EvaluationResult],
    *,
    failed_only: bool = False,
) -> dict[str, Any]:
    report_results = [
        result for result in results
        if not failed_only or not result.passed
    ]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "passed": sum(1 for result in results if result.passed),
        "total": len(results),
        "failed_only": failed_only,
        "results": [
            {
                "case": {
                    "id": result.case.id,
                    "question": result.case.question,
                },
                "passed": result.passed,
                "answer": result.answer,
                "tool_calls": _report_tool_calls(result.tool_calls),
                "failed_checks": [
                    asdict(check) for check in result.checks if not check.passed
                ],
                "failed_metrics": [
                    asdict(metric) for metric in result.metrics if not metric.passed
                ],
            }
            for result in report_results
        ],
    }


def _report_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact tool calls for reports without bulky tool results."""
    return [
        {
            "tool": call.get("tool"),
            "arguments": call.get("arguments", {}),
        }
        for call in tool_calls
    ]


if __name__ == "__main__":
    raise SystemExit(main())
