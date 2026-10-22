import json
from datetime import date

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from umai.agents.onrr_status_graph import try_answer_onrr_status_question


class OnrrStatusArguments(BaseModel):
    as_of_date: str
    status: str = "all"


class ProducingArguments(BaseModel):
    producing_date: str


def make_tools(calls):
    def count_wells_by_onrr_status(as_of_date, status="all"):
        calls.append(
            (
                "count_wells_by_onrr_status",
                {"as_of_date": as_of_date, "status": status},
            )
        )
        wells_by_status = {
            "active": [{"well": "Well - ALPHA"}, {"well": "Well - BRAVO"}],
            "inactive": [{"well": "Well - CHARLIE"}, {"well": "Well - DELTA"}],
            "injection": [{"well": "Well - INJECTOR"}],
        }
        return json.dumps(
            {
                "ok": True,
                "date": as_of_date,
                "status": status,
                "counts": {key: len(value) for key, value in wells_by_status.items()},
                "wells": wells_by_status.get(status),
            }
        )

    def get_producing_wells(producing_date):
        calls.append(("get_producing_wells", {"producing_date": producing_date}))
        return json.dumps(
            {
                "ok": True,
                "date": producing_date,
                "producing_count": 1,
                "producing_wells": [{"well": "Well - PRODUCER"}],
            }
        )

    return [
        StructuredTool.from_function(
            count_wells_by_onrr_status,
            name="count_wells_by_onrr_status",
            description="test",
            args_schema=OnrrStatusArguments,
        ),
        StructuredTool.from_function(
            get_producing_wells,
            name="get_producing_wells",
            description="test",
            args_schema=ProducingArguments,
        ),
    ]


def test_counts_inactive_wells_without_model_call():
    calls = []
    result = try_answer_onrr_status_question(
        tools=make_tools(calls),
        question="How many inactive wells yesterday?",
        today=date(2026, 7, 27),
    )

    assert result is not None
    answer, traces, stats = result
    assert answer == "There were 2 inactive wells on 07/26/2026."
    assert calls == [
        (
            "count_wells_by_onrr_status",
            {"as_of_date": "2026-07-26", "status": "inactive"},
        )
    ]
    assert traces[0]["tool"] == "count_wells_by_onrr_status"
    assert stats["model_calls"] == 0


def test_lists_inactive_wells_without_model_call():
    calls = []
    result = try_answer_onrr_status_question(
        tools=make_tools(calls),
        question="List inactive wells yesterday",
        today=date(2026, 7, 27),
    )

    assert result is not None
    answer, _, stats = result
    assert "Here are the 2 inactive wells on 07/26/2026:" in answer
    assert "- Well - CHARLIE" in answer
    assert "- Well - DELTA" in answer
    assert stats["model_calls"] == 0


def test_list_them_follow_up_uses_previous_status_and_date():
    calls = []
    result = try_answer_onrr_status_question(
        tools=make_tools(calls),
        question="list them",
        history=[
            {"role": "user", "content": "how many inactive wells yesterday?"},
            {
                "role": "assistant",
                "content": "There were 2 inactive wells on 07/26/2026.",
            },
        ],
        today=date(2026, 7, 27),
    )

    assert result is not None
    answer, _, stats = result
    assert calls == [
        (
            "count_wells_by_onrr_status",
            {"as_of_date": "2026-07-26", "status": "inactive"},
        )
    ]
    assert "Here are the 2 inactive wells on 07/26/2026:" in answer
    assert "- Well - CHARLIE" in answer
    assert stats["model_calls"] == 0


def test_list_them_without_status_history_falls_through():
    calls = []
    result = try_answer_onrr_status_question(
        tools=make_tools(calls),
        question="list them",
        history=[{"role": "user", "content": "How much oil did Battery 6 make?"}],
        today=date(2026, 7, 27),
    )

    assert result is None
    assert calls == []


def test_mixed_shutdown_status_question_falls_through_to_llm():
    calls = []
    result = try_answer_onrr_status_question(
        tools=make_tools(calls),
        question="How many active wells were shutdown yesterday?",
        today=date(2026, 7, 27),
    )

    assert result is None
    assert calls == []


def test_mixed_alarm_status_question_falls_through_to_llm():
    calls = []
    result = try_answer_onrr_status_question(
        tools=make_tools(calls),
        question="Which active wells had alarms yesterday?",
        today=date(2026, 7, 27),
    )

    assert result is None
    assert calls == []


def test_mixed_test_status_question_falls_through_to_llm():
    calls = []
    result = try_answer_onrr_status_question(
        tools=make_tools(calls),
        question="Which active wells had tests last month?",
        today=date(2026, 7, 27),
    )

    assert result is None
    assert calls == []


def test_follow_up_after_mixed_status_question_falls_through_to_llm():
    calls = []
    result = try_answer_onrr_status_question(
        tools=make_tools(calls),
        question="list them",
        history=[
            {
                "role": "user",
                "content": "How many active wells were shutdown yesterday?",
            },
            {"role": "assistant", "content": "There were 4 matching wells."},
        ],
        today=date(2026, 7, 27),
    )

    assert result is None
    assert calls == []


def test_producing_status_uses_producing_wells_tool():
    calls = []
    result = try_answer_onrr_status_question(
        tools=make_tools(calls),
        question="How many producing wells yesterday?",
        today=date(2026, 7, 27),
    )

    assert result is not None
    answer, _, stats = result
    assert calls == [("get_producing_wells", {"producing_date": "2026-07-26"})]
    assert answer == "There were 1 producing wells on 07/26/2026."
    assert stats["model_calls"] == 0
