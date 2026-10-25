import json
from pathlib import Path

from umai.config.settings import Settings
import pytest

from umai.evals.cases import EvaluationCaseError, filter_cases, load_cases
from umai.evals.checks import run_deterministic_checks
from umai.evals.models import EvaluationCase, EvaluationResult, MetricResult
from umai.evals.runner import (
    _configure_deepeval_environment,
    _evaluate_case,
    _report_payload,
    _write_report,
)
from umai.services.domain_guard import OUT_OF_DOMAIN_RESPONSE


def test_load_cases_parses_json_file(tmp_path: Path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "case-1",
                    "category": "reports",
                    "question": "How much gas was flared?",
                    "site_id": 4,
                    "deterministic": True,
                    "expected_output": "The answer should include the May 2026 gas flared total.",
                    "expected_tool_calls": [
                        {
                            "tool": "run_report",
                            "arguments": {"report_name": "gas_flared"},
                        }
                    ],
                    "metrics": ["answer_relevancy", "correctness"],
                    "metric_thresholds": {"answer_relevancy": 0.8},
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert len(cases) == 1
    assert cases[0].id == "case-1"
    assert cases[0].deterministic is True
    assert cases[0].expected_output == "The answer should include the May 2026 gas flared total."
    assert cases[0].expected_tool_calls == (
        {"tool": "run_report", "arguments": {"report_name": "gas_flared"}},
    )
    assert cases[0].metrics == ("answer_relevancy", "correctness")
    assert cases[0].metric_thresholds == {"answer_relevancy": 0.8}


def test_load_cases_rejects_invalid_expected_tool_calls(tmp_path: Path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "case-1",
                    "category": "reports",
                    "question": "How much gas was flared?",
                    "site_id": 4,
                    "expected_tool_calls": [{"arguments": {"report_name": "gas"}}],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationCaseError, match="must include tool"):
        load_cases(path)


def test_filter_cases_applies_category_case_id_and_limit():
    cases = [
        EvaluationCase(id="a", category="rag", question="q", site_id=4),
        EvaluationCase(id="b", category="reports", question="q", site_id=4),
        EvaluationCase(id="c", category="rag", question="q", site_id=4),
    ]

    assert [case.id for case in filter_cases(cases, category="rag")] == ["a", "c"]
    assert [case.id for case in filter_cases(cases, case_id="b")] == ["b"]
    assert [case.id for case in filter_cases(cases, limit=2)] == ["a", "b"]


def test_deterministic_checks_validate_tools_and_phrases():
    case = EvaluationCase(
        id="case-1",
        category="reports",
        question="q",
        site_id=4,
        expected_tool_calls=({"tool": "run_report", "arguments": {}},),
        forbidden_tools=("execute_operational_sql",),
        required_phrases=("gas flared",),
        forbidden_phrases=("site_id",),
    )

    checks = run_deterministic_checks(
        case,
        "Gas flared total was 123 barrels.",
        [{"tool": "run_report", "arguments": {}}],
    )

    assert all(check.passed for check in checks)


def test_deterministic_checks_validate_expected_tool_call_arguments():
    case = EvaluationCase(
        id="case-1",
        category="workflow_data_entry",
        question="q",
        site_id=4,
        expected_tool_calls=(
            {
                "tool": "prepare_data_entry",
                "arguments": {
                    "entry_type": "lact_reading",
                    "entity_name": {"contains_all": ["battery", "five", "lact"]},
                    "values.reading": 22,
                    "values.comments": {"contains": "hhh"},
                    "values.optional": {"absent": True},
                    "values.required": {"present": True},
                    "values.kind": {"one_of": ["oil", "lact"]},
                },
            },
        ),
    )

    checks = run_deterministic_checks(
        case,
        "Opening the prefilled data-entry form.",
        [
            {
                "tool": "prepare_data_entry",
                "arguments": {
                    "entry_type": "lact_reading",
                    "entity_name": "Battery five Lact",
                    "values": {
                        "reading": 22.0,
                        "comments": "hhhh",
                        "required": "yes",
                        "kind": "LACT",
                    },
                },
            }
        ],
    )

    assert all(check.passed for check in checks)


def test_deterministic_checks_validate_one_of_expected_tool_calls():
    case = EvaluationCase(
        id="case-1",
        category="readings",
        question="q",
        site_id=4,
        expected_tool_calls=(
            {
                "one_of": [
                    {
                        "tool": "search_readings",
                        "arguments": {
                            "reading_type": "well_fluid",
                            "entity_name": {"contains": "4048"},
                        },
                    },
                    {
                        "tool": "get_well_timeline",
                        "arguments": {
                            "well_name": {"contains": "4048"},
                            "context_query": {"contains_all": ["well", "fluid"]},
                        },
                    },
                ]
            },
        ),
    )

    checks = run_deterministic_checks(
        case,
        "Latest well fluids for 4048.",
        [
            {
                "tool": "get_well_timeline",
                "arguments": {
                    "well_name": "4048",
                    "start_date": "2026-04-03",
                    "end_date": "2026-07-03",
                    "context_query": "well fluid",
                },
            }
        ],
    )

    assert all(check.passed for check in checks)


def test_deterministic_checks_fail_expected_tool_call_argument_mismatch():
    case = EvaluationCase(
        id="case-1",
        category="reports",
        question="q",
        site_id=4,
        expected_tool_calls=(
            {
                "tool": "compare_report_periods",
                "arguments": {"first_start_date": "2026-06-01"},
            },
        ),
    )

    checks = run_deterministic_checks(
        case,
        "Compared production.",
        [
            {
                "tool": "compare_report_periods",
                "arguments": {"first_start_date": "2026-05-01"},
            }
        ],
    )

    assert any(
        check.name == "expected_tool_call:1:compare_report_periods"
        and not check.passed
        and "first_start_date" in check.message
        for check in checks
    )


def test_deterministic_checks_fail_missing_expected_tool_call():
    case = EvaluationCase(
        id="case-1",
        category="reports",
        question="q",
        site_id=4,
        expected_tool_calls=(
            {"tool": "compare_report_periods", "arguments": {"report_name": "oil"}},
        ),
    )

    checks = run_deterministic_checks(
        case,
        "Answer.",
        [{"tool": "run_report", "arguments": {"report_name": "oil"}}],
    )

    assert any(
        check.name == "expected_tool_call:1:compare_report_periods"
        and not check.passed
        and "was not used" in check.message
        for check in checks
    )


def test_deterministic_checks_reject_guardrail_answer_that_is_not_refusal():
    case = EvaluationCase(
        id="guardrail",
        category="guardrail",
        question="How old is Paris?",
        site_id=4,
    )

    checks = run_deterministic_checks(case, "Paris is very old.", [])

    assert any(
        check.name == "out_of_domain_refusal" and not check.passed
        for check in checks
    )


def test_deterministic_checks_accept_guardrail_refusal():
    case = EvaluationCase(
        id="guardrail",
        category="guardrail",
        question="How old is Paris?",
        site_id=4,
    )

    checks = run_deterministic_checks(case, OUT_OF_DOMAIN_RESPONSE, [])

    assert all(check.passed for check in checks)


def test_write_report_outputs_json(tmp_path: Path):
    result = EvaluationResult(
        case=EvaluationCase(id="case-1", category="reports", question="q", site_id=4),
        answer="answer",
        tool_calls=[
            {
                "tool": "run_report",
                "arguments": {"report_name": "oil_production"},
                "result": "{\"large\":\"payload\"}",
            }
        ],
        stats={"total_seconds": 1.0},
        checks=[],
        metrics=[
            MetricResult(name="answer_relevancy", score=1.0, passed=True),
            MetricResult(
                name="correctness",
                score=0.2,
                passed=False,
                reason="Missing expected output.",
            ),
        ],
    )

    path = _write_report([result], tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name.startswith("umai-eval-")
    assert payload["passed"] == 0
    assert payload["total"] == 1
    assert payload["results"][0]["case"] == {"id": "case-1", "question": "q"}
    assert "stats" not in payload["results"][0]
    assert "checks" not in payload["results"][0]
    assert "metrics" not in payload["results"][0]
    assert payload["results"][0]["tool_calls"] == [
        {
            "tool": "run_report",
            "arguments": {"report_name": "oil_production"},
        }
    ]
    assert payload["results"][0]["failed_checks"] == []
    assert payload["results"][0]["failed_metrics"] == [
        {
            "name": "correctness",
            "score": 0.2,
            "passed": False,
            "reason": "Missing expected output.",
            "error": None,
        }
    ]


def test_report_payload_marks_failed_results():
    result = EvaluationResult(
        case=EvaluationCase(id="case-1", category="reports", question="q", site_id=4),
        answer="site_id 4",
        tool_calls=[],
        stats={},
        checks=run_deterministic_checks(
            EvaluationCase(id="case-1", category="reports", question="q", site_id=4),
            "site_id 4",
            [],
        ),
        metrics=[],
    )

    payload = _report_payload([result])

    assert payload["passed"] == 0
    assert payload["results"][0]["passed"] is False
    assert "checks" not in payload["results"][0]
    assert payload["results"][0]["failed_checks"]
    assert all(
        check["passed"] is False for check in payload["results"][0]["failed_checks"]
    )


def test_report_payload_can_include_only_failed_results():
    passing = EvaluationResult(
        case=EvaluationCase(id="pass-case", category="reports", question="ok", site_id=4),
        answer="answer",
        tool_calls=[],
        stats={},
        checks=[],
        metrics=[],
    )
    failing = EvaluationResult(
        case=EvaluationCase(id="fail-case", category="reports", question="bad", site_id=4),
        answer="site_id 4",
        tool_calls=[],
        stats={},
        checks=run_deterministic_checks(
            EvaluationCase(id="fail-case", category="reports", question="bad", site_id=4),
            "site_id 4",
            [],
        ),
        metrics=[],
    )

    payload = _report_payload([passing, failing], failed_only=True)

    assert payload["passed"] == 1
    assert payload["total"] == 2
    assert payload["failed_only"] is True
    assert [item["case"]["id"] for item in payload["results"]] == ["fail-case"]


def test_evaluate_case_skips_metrics_for_deterministic_case(monkeypatch):
    settings = make_settings()

    def fake_answer_chat(**_kwargs):
        return "answer", [{"tool": "run_report", "arguments": {}}], {}

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("DeepEval metrics should not run for deterministic cases")

    monkeypatch.setattr("umai.evals.runner.answer_chat", fake_answer_chat)
    monkeypatch.setattr("umai.evals.runner.run_deepeval_metrics", fail_if_called)

    result = _evaluate_case(
        settings,
        EvaluationCase(
            id="case-1",
            category="reports",
            question="q",
            site_id=4,
            deterministic=True,
        ),
        current_date_override=None,
    )

    assert result.metrics == []


def test_evaluate_case_runs_metrics_for_non_deterministic_case(monkeypatch):
    settings = make_settings()
    metric = MetricResult(name="answer_relevancy", score=1.0, passed=True)

    def fake_answer_chat(**_kwargs):
        return "answer", [{"tool": "run_report", "arguments": {}}], {}

    def fake_metrics(*_args, **_kwargs):
        return [metric]

    monkeypatch.setattr("umai.evals.runner.answer_chat", fake_answer_chat)
    monkeypatch.setattr("umai.evals.runner.run_deepeval_metrics", fake_metrics)

    result = _evaluate_case(
        settings,
        EvaluationCase(id="case-1", category="reports", question="q", site_id=4),
        current_date_override=None,
    )

    assert result.metrics == [metric]


def test_configure_deepeval_environment_uses_provider_neutral_settings(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    settings = make_settings()

    _configure_deepeval_environment(settings)

    assert settings.llm_api_key
    assert settings.llm_base_url
    assert __import__("os").environ["OPENAI_API_KEY"] == settings.llm_api_key
    assert __import__("os").environ["OPENAI_BASE_URL"] == settings.llm_base_url
    assert __import__("os").environ["OPENAI_API_BASE"] == settings.llm_base_url


def make_settings() -> Settings:
    return Settings(
        llm_api_key="test-key",
        llm_model="gpt-5-mini",
        llm_base_url="https://api.openai.com/v1",
        umreports_api_url="http://127.0.0.1:50008/report/",
        umreports_timeout_seconds=30,
        max_report_days=366,
        db_host="127.0.0.1",
        db_port=3306,
        db_user="user",
        db_password="pass",
        db_name="ometrics",
        db_pool_size=5,
        db_max_overflow=5,
        db_pool_timeout=15,
        db_pool_recycle=1800,
        operational_sql_db_user="readonly",
        operational_sql_db_password="readonly-pass",
        operational_sql_max_rows=100,
        operational_sql_timeout_seconds=10,
        umai_max_concurrent=2,
        umai_slot_timeout=1,
        umai_conversation_history_limit=20,
        umai_conversation_ttl_hours=168,
        umai_daily_user_limit_enabled=True,
        umai_daily_user_limit_requests=100,
        timezone="UTC",
        vector_db_host="127.0.0.1",
        vector_db_port=5432,
        vector_db_user="ometrics",
        vector_db_password="",
        vector_db_name="ometrics",
        vector_db_pool_size=5,
        vector_db_max_overflow=5,
        vector_db_pool_timeout=15,
        vector_db_pool_recycle=1800,
        rag_embedding_model="text-embedding-3-small",
        rag_embedding_dimensions=1536,
        log_level="INFO",
    )
