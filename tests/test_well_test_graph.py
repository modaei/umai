import json
from datetime import date

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from umai.agents.well_test_graph import (
    prepare_well_test_analysis_dependency,
    try_answer_well_test_analysis,
)


class AnalysisArguments(BaseModel):
    analysis_mode: str
    group_by: str
    start_date: str | None = None
    end_date: str | None = None
    well_name: str | None = None
    battery_name: str | None = None
    test_count: int | None = None


def make_tool(calls, *, fail=False):
    def analyze_well_tests(**arguments):
        calls.append(arguments)
        if fail:
            return json.dumps({"ok": False, "error": "Database unavailable."})
        if arguments["analysis_mode"] == "range_summary":
            return json.dumps(
                {
                    "ok": True,
                    "analysis_mode": "range_summary",
                    "group_by": arguments["group_by"],
                    "start_date": arguments["start_date"],
                    "end_date": arguments["end_date"],
                    "groups": [
                        {
                            "group_name": "Battery 6",
                            "test_count": 4,
                            "well_count": 3,
                            "oil_sum": 42,
                            "oil_average": 10.5,
                            "water_sum": 80,
                            "water_average": 20,
                            "gas_sum": 8,
                            "gas_average": 2,
                        }
                    ],
                }
            )
        if arguments["analysis_mode"] in {"recent_tests", "range_sequence"}:
            return json.dumps(
                {
                    "ok": True,
                    "analysis_mode": arguments["analysis_mode"],
                    "group_by": "well",
                    "start_date": arguments.get("start_date"),
                    "end_date": arguments.get("end_date"),
                    "requested_test_count": arguments.get("test_count"),
                    "truncated": False,
                    "groups": [
                        {
                            "group_name": "HDU 4048",
                            "well_name": "HDU 4048",
                            "test_count": 2,
                            "tests": [
                                {
                                    "date": "2026-06-01",
                                    "oil": 10,
                                    "water": 20,
                                    "gas": 2,
                                    "runtime": 24,
                                    "oil_change": None,
                                    "water_change": None,
                                    "gas_change": None,
                                    "runtime_change": None,
                                },
                                {
                                    "date": "2026-06-15",
                                    "oil": 12,
                                    "water": 18,
                                    "gas": 3,
                                    "runtime": 20,
                                    "oil_change": 2,
                                    "water_change": -2,
                                    "gas_change": 1,
                                    "runtime_change": -4,
                                },
                            ],
                        }
                    ],
                }
            )
        return json.dumps(
            {
                "ok": True,
                "analysis_mode": "latest_previous",
                "group_by": arguments["group_by"],
                "groups": [
                    {
                        "group_name": "HDU 4048",
                        "previous_date": "2026-06-01",
                        "latest_date": "2026-06-15",
                        "previous_oil": 10,
                        "latest_oil": 12,
                        "oil_change": 2,
                        "previous_water": 20,
                        "latest_water": 18,
                        "water_change": -2,
                        "previous_gas": 2,
                        "latest_gas": 3,
                        "gas_change": 1,
                        "previous_runtime": 24,
                        "latest_runtime": 20,
                        "runtime_change": -4,
                    }
                ],
            }
        )

    return StructuredTool.from_function(
        analyze_well_tests,
        name="analyze_well_tests",
        description="test",
        args_schema=AnalysisArguments,
    )


def test_latest_previous_analysis_is_terminal_and_skips_model():
    calls = []
    result = try_answer_well_test_analysis(
        tools=[make_tool(calls)],
        question="For each well, compare latest well test and last test.",
        history=[],
        today=date(2026, 7, 2),
    )

    assert result is not None
    answer, traces, stats = result
    assert calls[0]["analysis_mode"] == "latest_previous"
    assert calls[0]["group_by"] == "well"
    assert "HDU 4048" in answer
    assert "10 -> 12 (+2) bbl" in answer
    assert traces[0]["tool"] == "analyze_well_tests"
    assert stats["model_calls"] == 0


def test_range_summary_routes_by_battery_with_dates():
    calls = []
    result = try_answer_well_test_analysis(
        tools=[make_tool(calls)],
        question="Summarize well tests by battery last month.",
        history=[],
        today=date(2026, 7, 2),
    )

    assert result is not None
    assert calls[0]["analysis_mode"] == "range_summary"
    assert calls[0]["group_by"] == "battery"
    assert calls[0]["start_date"] == "2026-06-01"
    assert calls[0]["end_date"] == "2026-06-30"
    assert "Battery 6" in result[0]


def test_follow_up_uses_previous_well_test_period_and_typo():
    calls = []
    result = try_answer_well_test_analysis(
        tools=[make_tool(calls)],
        question="break down please and compare each test with previus",
        history=[
            {"role": "user", "content": "what are last month well tests"},
            {"role": "assistant", "content": "There were 88 well tests."},
        ],
        today=date(2026, 7, 2),
    )

    assert result is not None
    assert calls[0]["analysis_mode"] == "latest_previous"
    assert calls[0]["start_date"] == "2026-06-01"
    assert calls[0]["end_date"] == "2026-06-30"


def test_composite_request_prefetches_dependency_instead_of_terminal_answer():
    calls = []
    tools = [make_tool(calls)]
    terminal = try_answer_well_test_analysis(
        tools=tools,
        question="Compare latest and previous well tests and explain shutdown causes.",
        history=[],
        today=date(2026, 7, 2),
    )
    dependency = prepare_well_test_analysis_dependency(
        tools=tools,
        question="Compare latest and previous well tests and explain shutdown causes.",
        history=[],
        today=date(2026, 7, 2),
    )

    assert terminal is None
    assert dependency is not None
    context = json.loads(dependency["context"])
    assert context["authoritative_data_type"] == "well_test_analysis"
    assert context["result"]["groups"][0]["group_name"] == "HDU 4048"
    assert dependency["tool_name"] == "analyze_well_tests"


def test_terminal_tool_failure_returns_controlled_error():
    calls = []
    result = try_answer_well_test_analysis(
        tools=[make_tool(calls, fail=True)],
        question="Compare each well's latest and previous well test.",
        history=[],
        today=date(2026, 7, 2),
    )

    assert result is not None
    assert result[0] == "Database unavailable."
    assert result[2]["model_calls"] == 0


def test_raw_well_test_search_falls_through():
    calls = []
    result = try_answer_well_test_analysis(
        tools=[make_tool(calls)],
        question="Show well tests with oil greater than 20 in June 2026.",
        history=[],
        today=date(2026, 7, 2),
    )

    assert result is None
    assert calls == []


def test_latest_three_tests_by_battery_routes_without_model():
    calls = []
    result = try_answer_well_test_analysis(
        tools=[make_tool(calls)],
        question="For wells in battery 6 compare last three well tests.",
        history=[],
        today=date(2026, 7, 2),
    )

    assert result is not None
    assert calls[0]["analysis_mode"] == "recent_tests"
    assert calls[0]["group_by"] == "well"
    assert calls[0]["test_count"] == 3
    assert calls[0]["battery_name"] == "Battery 6"
    assert "baseline (no preceding selected test)" in result[0]
    assert "oil 12 bbl (+2)" in result[0]
    assert result[2]["model_calls"] == 0


def test_latest_three_tests_extracts_bare_numeric_well_filter():
    calls = []
    result = try_answer_well_test_analysis(
        tools=[make_tool(calls)],
        question="Compare last three well test for 4048",
        history=[],
        today=date(2026, 7, 2),
    )

    assert result is not None
    assert calls[0]["analysis_mode"] == "recent_tests"
    assert calls[0]["group_by"] == "well"
    assert calls[0]["test_count"] == 3
    assert calls[0]["well_name"] == "4048"


def test_latest_three_tests_extracts_hdu_shorthand_well_filters():
    for question in (
        "Compare last three well test for hdu_4048",
        "Compare last three well test for hdu 4048",
        "Compare last three well test for hdu-4048",
        "For well 4048, compare last three well tests",
        "Compare last three well tests for HARTZOG DRAW UNIT 4048",
    ):
        calls = []
        result = try_answer_well_test_analysis(
            tools=[make_tool(calls)],
            question=question,
            history=[],
            today=date(2026, 7, 2),
        )

        assert result is not None
        assert calls[0]["analysis_mode"] == "recent_tests"
        assert calls[0]["group_by"] == "well"
        assert calls[0]["test_count"] == 3
        assert calls[0]["well_name"] == "4048"


def test_latest_three_tests_on_all_wells_remains_unscoped():
    calls = []
    result = try_answer_well_test_analysis(
        tools=[make_tool(calls)],
        question="Are there any anomalies between last three well test on all wells",
        history=[],
        today=date(2026, 7, 2),
    )

    assert result is not None
    assert calls[0]["analysis_mode"] == "recent_tests"
    assert calls[0]["group_by"] == "well"
    assert calls[0]["test_count"] == 3
    assert calls[0].get("well_name") is None


def test_compare_tests_in_month_routes_to_per_well_range_sequence():
    calls = []
    result = try_answer_well_test_analysis(
        tools=[make_tool(calls)],
        question="For wells in battery 6 compare well tests in June 2026.",
        history=[],
        today=date(2026, 7, 2),
    )

    assert result is not None
    assert calls[0]["analysis_mode"] == "range_sequence"
    assert calls[0]["group_by"] == "well"
    assert calls[0]["start_date"] == "2026-06-01"
    assert calls[0]["end_date"] == "2026-06-30"
    assert calls[0]["battery_name"] == "Battery 6"
    assert "06/01/2026 through 06/30/2026" in result[0]
