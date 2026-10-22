import json
from datetime import date

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from umai.agents.producing_wells_graph import (
    prepare_well_population_dependency,
    try_answer_producing_well_question,
)


class AnyArguments(BaseModel):
    producing_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    range_mode: str = "any_day"
    filters: list[dict] = []


class WellTestArguments(BaseModel):
    start_date: str
    end_date: str


class ActiveArguments(BaseModel):
    active_date: str


def make_tools(calls):
    def get_producing_wells(**arguments):
        calls.append(("get_producing_wells", arguments))
        if arguments.get("start_date") == "2026-06-03":
            producing_wells = [
                {"well": f"Well - WELL {index}"} for index in range(1, 9)
            ]
        else:
            producing_wells = [
                {"well": "Well - ALPHA"},
                {"well": "Well - BRAVO"},
                {"well": "Well - CHARLIE"},
            ]
        return json.dumps(
            {
                "ok": True,
                "producing_count": len(producing_wells),
                "producing_wells": producing_wells,
            }
        )

    def search_well_tests(start_date, end_date):
        calls.append(
            ("search_well_tests", {"start_date": start_date, "end_date": end_date})
        )
        well_tests = (
            [{"entity_display_name": "Well - WELL 1"}]
            if start_date == "2026-06-03"
            else [
                {"entity_display_name": "Well - ALPHA"},
                {"entity_display_name": "Well - CHARLIE"},
            ]
        )
        return json.dumps(
            {
                "ok": True,
                "well_tests": well_tests,
            }
        )

    def get_active_wells(active_date):
        calls.append(("get_active_wells", {"active_date": active_date}))
        return json.dumps(
            {
                "ok": True,
                "date": active_date,
                "active_count": 2,
                "active_wells": [
                    {"well": "Well - ALPHA"},
                    {"well": "Well - BRAVO"},
                ],
                "rules": {
                    "active_well_rule": (
                        "Active wells must have active_well=true on the ONRR code "
                        "as of that day."
                    )
                },
            }
        )
    return [
        StructuredTool.from_function(
            get_producing_wells,
            name="get_producing_wells",
            description="test",
            args_schema=AnyArguments,
        ),
        StructuredTool.from_function(
            search_well_tests,
            name="search_well_tests",
            description="test",
            args_schema=WellTestArguments,
        ),
        StructuredTool.from_function(
            get_active_wells,
            name="get_active_wells",
            description="test",
            args_schema=ActiveArguments,
        ),
    ]


def test_routes_producer_well_test_coverage_without_model_call():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="Which producing wells had no well test in June 2026?",
        today=date(2026, 7, 2),
    )

    assert result is not None
    answer, traces, stats = result
    assert calls[0][0] == "get_producing_wells"
    assert calls[0][1]["start_date"] == "2026-06-01"
    assert calls[0][1]["end_date"] == "2026-06-30"
    assert calls[0][1]["range_mode"] == "any_day"
    assert calls[1][0] == "search_well_tests"
    assert "BRAVO" in answer
    assert "ALPHA" not in answer
    assert [trace["tool"] for trace in traces] == [
        "get_producing_wells",
        "search_well_tests",
    ]
    assert stats["model_calls"] == 0
    assert stats["model_seconds"] == 0


def test_routes_active_well_test_coverage_without_model_call():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="Which active wells don't have a test in past 30 days?",
        today=date(2026, 7, 27),
    )

    assert result is not None
    answer, traces, stats = result
    assert calls == [
        ("get_active_wells", {"active_date": "2026-07-27"}),
        (
            "search_well_tests",
            {"start_date": "2026-06-28", "end_date": "2026-07-27"},
        ),
    ]
    assert "As of 07/27/2026, 1 of 2 active wells had no well test" in answer
    assert "Active means ONRR-active as of 07/27/2026." in answer
    assert "BRAVO" in answer
    assert "ALPHA" not in answer
    assert [trace["tool"] for trace in traces] == [
        "get_active_wells",
        "search_well_tests",
    ]
    assert stats["model_calls"] == 0
    assert stats["model_seconds"] == 0


def test_routes_active_well_test_coverage_with_explicit_active_date():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="Which active wells on July 1 don't have tests in past 30 days?",
        today=date(2026, 7, 27),
    )

    assert result is not None
    answer, _, stats = result
    assert calls[0] == ("get_active_wells", {"active_date": "2026-07-01"})
    assert calls[1] == (
        "search_well_tests",
        {"start_date": "2026-06-28", "end_date": "2026-07-27"},
    )
    assert "As of 07/01/2026" in answer
    assert stats["model_calls"] == 0


def test_routes_active_well_test_coverage_count_only():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="How many active wells don't have a test in past 30 days?",
        today=date(2026, 7, 27),
    )

    assert result is not None
    answer, _, stats = result
    assert "1 of 2 active wells had no well test" in answer
    assert "BRAVO" not in answer
    assert stats["model_calls"] == 0


def test_routes_simple_current_producer_count_to_single_date_tool():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="How many producing wells do we have today?",
        today=date(2026, 7, 2),
    )

    assert result is not None
    answer, _, stats = result
    assert calls[0][0] == "get_producing_wells"
    assert calls[0][1]["producing_date"] == "2026-07-02"
    assert answer == "There were 3 producing wells on 07/02/2026."
    assert stats["model_calls"] == 0


def test_routes_battery_filtered_producer_count_to_single_date_tool():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="How many producing wells were in Battery 6 yesterday?",
        today=date(2026, 7, 9),
    )

    assert result is not None
    answer, _, stats = result
    assert calls[0][0] == "get_producing_wells"
    assert calls[0][1]["producing_date"] == "2026-07-08"
    assert calls[0][1]["filters"] == [{"field": "battery", "value": "Battery 6"}]
    assert answer == "There were 3 producing wells in Battery 6 on 07/08/2026."
    assert stats["model_calls"] == 0


def test_routes_pump_type_filtered_producer_count_to_single_date_tool():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="How many rod producing wells were there yesterday?",
        today=date(2026, 7, 9),
    )

    assert result is not None
    answer, _, _ = result
    assert calls[0][1]["producing_date"] == "2026-07-08"
    assert calls[0][1]["filters"] == [{"field": "pump_type", "value": "rod"}]
    assert "with pump type ROD" in answer


def test_unrelated_question_falls_through_to_general_agent():
    calls = []

    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="How much oil did Battery 6 produce in June?",
        today=date(2026, 7, 2),
    )

    assert result is None
    assert calls == []


def test_active_well_range_count_falls_through_to_general_agent():
    calls = []

    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="How many active wells were there in June 2026?",
        today=date(2026, 7, 2),
    )

    assert result is None
    assert calls == []


def test_prefetches_producing_population_for_broader_question():
    calls = []

    dependency = prepare_well_population_dependency(
        tools=make_tools(calls),
        question="Summarize alarms for producing wells in June 2026.",
        today=date(2026, 7, 2),
    )

    assert dependency is not None
    assert dependency["tool_name"] == "get_producing_wells"
    context = json.loads(dependency["context"])
    assert context["authoritative_population"] == "producing_wells"
    assert context["well_names"] == ["ALPHA", "BRAVO", "CHARLIE"]
    assert dependency["stats"]["model_calls"] == 0


def test_prefetches_active_population_for_single_day_broader_question():
    calls = []

    dependency = prepare_well_population_dependency(
        tools=make_tools(calls),
        question="Which active wells had alarms yesterday?",
        today=date(2026, 7, 2),
    )

    assert dependency is not None
    assert calls == [("get_active_wells", {"active_date": "2026-07-01"})]
    context = json.loads(dependency["context"])
    assert context["well_names"] == ["ALPHA", "BRAVO"]
    assert context["partial_shutdown_well_names"] == []


def test_does_not_apply_undefined_active_well_range_semantics():
    calls = []

    dependency = prepare_well_population_dependency(
        tools=make_tools(calls),
        question="Which active wells had alarms in June 2026?",
        today=date(2026, 7, 2),
    )

    assert dependency is None
    assert calls == []


def test_routes_last_sixty_days_coverage_without_model_call():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question=(
            "For last 60 days which active oil producing well does not have "
            "at least one test?"
        ),
        history=[],
        today=date(2026, 7, 2),
    )

    assert result is not None
    answer, _, stats = result
    assert calls[0][1]["start_date"] == "2026-05-04"
    assert calls[0][1]["end_date"] == "2026-07-02"
    assert calls[1] == (
        "search_well_tests",
        {"start_date": "2026-05-04", "end_date": "2026-07-02"},
    )
    assert "BRAVO" in answer
    assert stats["model_calls"] == 0


def test_follow_up_replaces_coverage_range_from_history():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="How about last 30 days?",
        history=[
            {
                "role": "user",
                "content": (
                    "For last 60 days which active oil producing well does not "
                    "have at least one test?"
                ),
            },
            {"role": "assistant", "content": "One well was missing a test."},
        ],
        today=date(2026, 7, 2),
    )

    assert result is not None
    answer, _, stats = result
    assert calls[0][1]["start_date"] == "2026-06-03"
    assert calls[0][1]["end_date"] == "2026-07-02"
    assert calls[1][1] == {"start_date": "2026-06-03", "end_date": "2026-07-02"}
    assert "7 of 8" in answer
    assert "WELL 8" in answer
    assert stats["model_calls"] == 0


def test_last_one_day_is_inclusive_of_today():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="Which producing wells had no well test in the last 1 day?",
        today=date(2026, 7, 2),
    )

    assert result is not None
    assert calls[0][1]["producing_date"] == "2026-07-02"
    assert calls[1][1] == {"start_date": "2026-07-02", "end_date": "2026-07-02"}


def test_count_only_follow_up_reuses_coverage_range():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="Count only.",
        history=[
            {
                "role": "user",
                "content": "Which producing wells had no well test in June 2026?",
            }
        ],
        today=date(2026, 7, 2),
    )

    assert result is not None
    assert "BRAVO" not in result[0]
    assert "1 of 3" in result[0]


def test_unrelated_follow_up_does_not_reuse_coverage_intent():
    calls = []
    result = try_answer_producing_well_question(
        tools=make_tools(calls),
        question="What about shutdowns in the last 30 days?",
        history=[
            {
                "role": "user",
                "content": "Which producing wells had no well test in June 2026?",
            }
        ],
        today=date(2026, 7, 2),
    )

    assert result is None
    assert calls == []
