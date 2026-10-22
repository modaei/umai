import json
from datetime import date

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from umai.agents.production_context_graph import prepare_production_context_dependency


class MonthlyReportArguments(BaseModel):
    report_name: str
    year: int
    battery_name: str | None = None
    value_key: str | None = None


class OperationalContextArguments(BaseModel):
    query: str
    start_date: str | None = None
    end_date: str | None = None
    entity_name: str | None = None
    limit: int = 8


def make_tools(calls, matches_by_entity=None):
    def summarize_report_by_month(**arguments):
        calls.append(("summarize_report_by_month", arguments))
        return json.dumps(
            {
                "ok": True,
                "report_name": arguments["report_name"],
                "year": arguments["year"],
                "battery_name": arguments["battery_name"],
                "months": [{"month": "2026-06", "total": 4314}],
            }
        )

    def search_operational_context(**arguments):
        calls.append(("search_operational_context", arguments))
        matches = (
            matches_by_entity.get(arguments.get("entity_name"), [])
            if matches_by_entity is not None
            else [{"text": "Battery 6 had emulsion correction notes."}]
        )
        return json.dumps(
            {
                "ok": True,
                "count": len(matches),
                "matches": matches,
            }
        )

    return [
        StructuredTool.from_function(
            summarize_report_by_month,
            name="summarize_report_by_month",
            description="test",
            args_schema=MonthlyReportArguments,
        ),
        StructuredTool.from_function(
            search_operational_context,
            name="search_operational_context",
            description="test",
            args_schema=OperationalContextArguments,
        ),
    ]


class FakeWellFilterClient:
    def find_wells(self, site_id, well_name=None, filters=None):
        assert site_id == 4
        assert filters == [{"field": "battery", "value": "Battery 6"}]
        return {
            "wells": [
                {"name": "HARTZOG DRAW UNIT 4048"},
                {"name": "HARTZOG DRAW UNIT 5823"},
            ]
        }


class FakeReadingClient:
    def list_equipment(self, site_id, equipment_type, battery_name=None, **kwargs):
        assert site_id == 4
        assert battery_name == "Battery 6"
        if equipment_type == "tank":
            return {"entities": [{"entity_name": "5-2 Float Over"}]}
        if equipment_type == "pump":
            return {"entities": [{"entity_name": "Battery 6 Pump"}]}
        return {"entities": []}


def test_prefetches_production_variation_report_and_context():
    calls = []

    dependency = prepare_production_context_dependency(
        tools=make_tools(calls),
        question="why battery 6 had oil production variation in 2026?",
        site_id=4,
        reading_client=FakeReadingClient(),
        well_filter_client=FakeWellFilterClient(),
    )

    assert dependency is not None
    assert calls[0] == (
        "summarize_report_by_month",
        {
            "report_name": "oil_production",
            "year": 2026,
            "battery_name": "Battery 6",
            "value_key": None,
        },
    )
    assert calls[1] == (
        "search_operational_context",
        {
            "query": "Battery 6 oil production variation reason 2026",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "entity_name": "Battery 6",
            "limit": 10,
        },
    )
    assert calls[2][0] == "search_operational_context"
    assert calls[2][1]["entity_name"] == "HARTZOG DRAW UNIT 4048"
    assert calls[3][1]["entity_name"] == "HARTZOG DRAW UNIT 5823"
    assert calls[4][1]["entity_name"] == "5-2 Float Over"
    assert calls[5][1]["entity_name"] == "Battery 6 Pump"
    context = json.loads(dependency["context"])
    assert context["authoritative_data_type"] == "production_variation_context"
    assert context["related_entities"]["wells"] == [
        "HARTZOG DRAW UNIT 4048",
        "HARTZOG DRAW UNIT 5823",
    ]
    assert context["related_entities"]["equipment"]["tank"] == ["5-2 Float Over"]
    assert context["related_entities"]["equipment"]["pump"] == ["Battery 6 Pump"]
    assert [trace["tool"] for trace in dependency["traces"]] == [
        "summarize_report_by_month",
        "search_operational_context",
        "search_operational_context",
        "search_operational_context",
        "search_operational_context",
        "search_operational_context",
    ]
    assert dependency["stats"]["model_calls"] == 0


def test_prefetches_month_comparison_reason_with_inferred_year():
    calls = []

    dependency = prepare_production_context_dependency(
        tools=make_tools(calls),
        question="Why was Battery 6 oil production lower in June than in May?",
        site_id=4,
        today=date(2026, 7, 10),
        reading_client=FakeReadingClient(),
        well_filter_client=FakeWellFilterClient(),
        max_related_entities=1,
    )

    assert dependency is not None
    assert calls[0] == (
        "summarize_report_by_month",
        {
            "report_name": "oil_production",
            "year": 2026,
            "battery_name": "Battery 6",
            "value_key": None,
        },
    )
    assert calls[1] == (
        "search_operational_context",
        {
            "query": "Battery 6 oil production lower June May reason 2026",
            "start_date": "2026-01-01",
            "end_date": "2026-12-31",
            "entity_name": "Battery 6",
            "limit": 10,
        },
    )
    context = json.loads(dependency["context"])
    assert context["focus_periods"] == ["2026-06", "2026-05"]


def test_short_reason_follow_up_uses_previous_battery_comparison():
    calls = []

    dependency = prepare_production_context_dependency(
        tools=make_tools(calls),
        question="why?",
        history=[
            {
                "role": "user",
                "content": "Compare Battery 6 oil production in June and May 2026",
            },
            {
                "role": "assistant",
                "content": (
                    "Battery 6 produced 3,849.36 bbl in June 2026 and "
                    "4,165.64 bbl in May 2026. June was down 7.6%."
                ),
            },
        ],
        site_id=4,
        today=date(2026, 7, 10),
        reading_client=FakeReadingClient(),
        well_filter_client=FakeWellFilterClient(),
    )

    assert dependency is not None
    assert calls[0] == (
        "summarize_report_by_month",
        {
            "report_name": "oil_production",
            "year": 2026,
            "battery_name": "Battery 6",
            "value_key": None,
        },
    )
    searched_entities = [
        arguments["entity_name"]
        for tool_name, arguments in calls
        if tool_name == "search_operational_context"
    ]
    assert searched_entities == [
        "Battery 6",
        "HARTZOG DRAW UNIT 4048",
        "HARTZOG DRAW UNIT 5823",
        "5-2 Float Over",
        "Battery 6 Pump",
    ]
    context = json.loads(dependency["context"])
    assert context["focus_periods"] == ["2026-06", "2026-05"]


def test_short_reason_follow_up_without_production_history_does_not_prefetch():
    dependency = prepare_production_context_dependency(
        tools=make_tools([]),
        question="why?",
        history=[
            {"role": "user", "content": "How old is Paris?"},
            {"role": "assistant", "content": "I can only help with Ometrics."},
        ],
        site_id=4,
        today=date(2026, 7, 10),
    )

    assert dependency is None


def test_filters_operational_context_to_verified_battery_scope():
    calls = []
    matches_by_entity = {
        "Battery 6": [
            {
                "chunk_id": "general_note:valid-battery",
                "entity_name": None,
                "text": "Battery 6 low production due to hot oil well.",
            },
            {
                "chunk_id": "general_note:invalid-3051",
                "entity_name": None,
                "text": (
                    "Working with accounting to fix negative production due to "
                    "emulsion issue found at 3051 where water was counted as oil."
                ),
            },
        ],
        "HARTZOG DRAW UNIT 4048": [
            {
                "chunk_id": "well_shutdown:valid-full-name",
                "entity_name": "HARTZOG DRAW UNIT 4048",
                "text": "Well shutdown. Well Name: HARTZOG DRAW UNIT 4048 down 4 hrs.",
            },
            {
                "chunk_id": "well_shutdown:invalid-3051-entity",
                "entity_name": "HARTZOG DRAW UNIT 3051",
                "text": "Well shutdown. Well Name: HARTZOG DRAW UNIT 3051 down 8 hrs.",
            },
        ],
        "HARTZOG DRAW UNIT 5823": [
            {
                "chunk_id": "general_note:valid-short-well",
                "entity_name": None,
                "text": "5823 was down for hot oil work.",
            },
        ],
        "5-2 Float Over": [
            {
                "chunk_id": "work_order:valid-equipment",
                "entity_name": "5-2 Float Over repair",
                "text": "Work order for 5-2 Float Over.",
            },
        ],
        "Battery 6 Pump": [
            {
                "chunk_id": "work_order:valid-pump-text",
                "entity_name": None,
                "text": "Battery 6 Pump had maintenance.",
            },
        ],
    }

    dependency = prepare_production_context_dependency(
        tools=make_tools(calls, matches_by_entity=matches_by_entity),
        question="Why was Battery 6 oil production lower in June than in May?",
        site_id=4,
        today=date(2026, 7, 10),
        reading_client=FakeReadingClient(),
        well_filter_client=FakeWellFilterClient(),
    )

    assert dependency is not None
    context = json.loads(dependency["context"])
    results = context["operational_context_by_entity"]["results"]
    kept_chunk_ids = [
        match["chunk_id"]
        for result in results
        for match in result["result"]["matches"]
    ]

    assert kept_chunk_ids == [
        "general_note:valid-battery",
        "well_shutdown:valid-full-name",
        "general_note:valid-short-well",
        "work_order:valid-equipment",
        "work_order:valid-pump-text",
    ]
    assert "general_note:invalid-3051" not in json.dumps(context)
    assert "well_shutdown:invalid-3051-entity" not in json.dumps(context)
    assert "general_note:invalid-3051" not in json.dumps(dependency["traces"])
    assert "well_shutdown:invalid-3051-entity" not in json.dumps(dependency["traces"])
    assert results[0]["result"]["original_count"] == 2
    assert results[0]["result"]["filtered_count"] == 1
    assert results[0]["result"]["excluded_count"] == 1


def test_month_comparison_without_reason_does_not_prefetch_production_context():
    dependency = prepare_production_context_dependency(
        tools=make_tools([]),
        question="Compare Battery 6 oil production in June and May",
        site_id=4,
        today=date(2026, 7, 10),
    )

    assert dependency is None


def test_non_reason_question_does_not_prefetch_production_context():
    dependency = prepare_production_context_dependency(
        tools=make_tools([]),
        question="show battery 6 oil production in 2026",
        site_id=4,
    )

    assert dependency is None
