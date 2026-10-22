import json
from datetime import date

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from umai.agents.report_comparison_graph import prepare_report_comparison_dependency


class CompareArguments(BaseModel):
    report_name: str
    first_start_date: str
    first_end_date: str
    second_start_date: str
    second_end_date: str


def make_tool(calls):
    def compare_report_periods(**arguments):
        calls.append(arguments)
        return json.dumps(
            {
                "ok": True,
                "report_name": arguments["report_name"],
                "first_period": {
                    "start_date": arguments["first_start_date"],
                    "end_date": arguments["first_end_date"],
                    "data": {"battery_totals": {"Battery 6": 12, "Battery 8": 5}},
                },
                "second_period": {
                    "start_date": arguments["second_start_date"],
                    "end_date": arguments["second_end_date"],
                    "data": {"battery_totals": {"Battery 6": 10, "Battery 8": 7}},
                },
            }
        )

    return StructuredTool.from_function(
        compare_report_periods,
        name="compare_report_periods",
        description="test",
        args_schema=CompareArguments,
    )


def test_report_comparison_resolves_yesterday_and_day_before():
    calls = []
    dependency = prepare_report_comparison_dependency(
        tools=[make_tool(calls)],
        question="compare each battery oil production between yesterday and the day before",
        today=date(2026, 7, 3),
    )

    assert dependency is not None
    assert calls[0] == {
        "report_name": "battery",
        "first_start_date": "2026-07-02",
        "first_end_date": "2026-07-02",
        "second_start_date": "2026-07-01",
        "second_end_date": "2026-07-01",
    }
    context = json.loads(dependency["context"])
    assert context["authoritative_data_type"] == "report_comparison"
    assert context["tool"] == "compare_report_periods"
    assert context["arguments"] == calls[0]
    assert dependency["tool_name"] == "compare_report_periods"
    assert dependency["stats"]["model_calls"] == 0


def test_report_comparison_resolves_last_month_and_month_before():
    calls = []
    dependency = prepare_report_comparison_dependency(
        tools=[make_tool(calls)],
        question="compare last month oil production with the month before and analyse the variation",
        today=date(2026, 6, 25),
    )

    assert dependency is not None
    assert calls[0] == {
        "report_name": "oil_production",
        "first_start_date": "2026-05-01",
        "first_end_date": "2026-05-31",
        "second_start_date": "2026-04-01",
        "second_end_date": "2026-04-30",
    }


def test_report_comparison_preserves_explicit_month_order():
    calls = []
    dependency = prepare_report_comparison_dependency(
        tools=[make_tool(calls)],
        question="compare oil production in June and May and analyze",
        today=date(2026, 7, 3),
    )

    assert dependency is not None
    assert calls[0] == {
        "report_name": "oil_production",
        "first_start_date": "2026-06-01",
        "first_end_date": "2026-06-30",
        "second_start_date": "2026-05-01",
        "second_end_date": "2026-05-31",
    }
