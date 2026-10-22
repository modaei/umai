import json
from datetime import date

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from umai.agents.navigation_router import try_answer_navigation_request


class ViewArguments(BaseModel):
    view_type: str
    start_date: str | None = None
    end_date: str | None = None
    search_text: str | None = None
    status: str | None = None


class EntryArguments(BaseModel):
    entry_type: str
    entity_name: str | None = None
    values: dict = Field(default_factory=dict)


class CapabilityArguments(BaseModel):
    query: str
    limit: int = 3


def make_tools(calls, *, view_status="ready", entry_status="ready"):
    def prepare_data_view(**arguments):
        calls.append(("prepare_data_view", arguments))
        return json.dumps(
            {
                "status": view_status,
                "destination": "index",
                "view_type": arguments["view_type"],
                "message": "Prepared view.",
            }
        )

    def prepare_data_entry(**arguments):
        calls.append(("prepare_data_entry", arguments))
        if entry_status == "needs_clarification":
            return json.dumps(
                {
                    "status": entry_status,
                    "message": "Which LACT should this use?",
                    "candidates": [
                        {"type": "LACT", "name": "LACT 1"},
                        {"type": "LACT", "name": "LACT 2"},
                    ],
                }
            )
        return json.dumps(
            {
                "status": entry_status,
                "entry_type": arguments["entry_type"],
            }
        )

    def search_ometrics_capabilities(**arguments):
        calls.append(("search_ometrics_capabilities", arguments))
        return json.dumps(
            {
                "ok": True,
                "query": arguments["query"],
                "count": 1,
                "matches": [
                    {
                        "capability": "LACT reading",
                        "summary": (
                            'For a LACT reading, say "Use the Create LACT readings form, '
                            'accessible from the dashboard or the Create LACT reading '
                            'button in the LACT readings page." For a Flare reading, say '
                            '"Use the Create Flare reading form, accessible from the '
                            'dashboard or the Create Flare reading button in the flare '
                            'readings page."'
                        ),
                    }
                ],
            }
        )

    return [
        StructuredTool.from_function(
            prepare_data_view,
            name="prepare_data_view",
            description="test",
            args_schema=ViewArguments,
        ),
        StructuredTool.from_function(
            prepare_data_entry,
            name="prepare_data_entry",
            description="test",
            args_schema=EntryArguments,
        ),
        StructuredTool.from_function(
            search_ometrics_capabilities,
            name="search_ometrics_capabilities",
            description="test",
            args_schema=CapabilityArguments,
        ),
    ]


def test_show_me_report_routes_with_relative_dates_without_model():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="Show me oil production this month",
        today=date(2026, 7, 3),
    )

    assert result is not None
    assert calls[0][0] == "prepare_data_view"
    assert calls[0][1]["view_type"] == "oil_production"
    assert calls[0][1]["start_date"] == "2026-07-01"
    assert calls[0][1]["end_date"] == "2026-07-03"
    assert result[2]["model_calls"] == 0
    assert json.loads(result[1][0]["result"])["status"] == "ready"


def test_how_do_i_register_lact_reading_routes_to_capability_help():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="How do I register a LACT reading?",
        today=date(2026, 7, 4),
    )

    assert result is not None
    assert calls[0][0] == "search_ometrics_capabilities"
    assert calls[0][1]["query"] == "How do I register a LACT reading?"
    assert "Create LACT readings" in result[0]
    assert "dashboard" in result[0]
    assert "Create LACT reading" in result[0]
    assert "LACT readings page" in result[0]
    assert result[2]["model_calls"] == 0


def test_where_can_i_enter_lact_reading_routes_to_capability_help():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="Where can I enter a LACT reading?",
        today=date(2026, 7, 4),
    )

    assert result is not None
    assert calls[0][0] == "search_ometrics_capabilities"
    assert "Create LACT readings" in result[0]


def test_how_do_i_register_flare_reading_uses_matching_capability_guidance():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="How do I register a flare reading?",
        today=date(2026, 7, 4),
    )

    assert result is not None
    assert calls[0][0] == "search_ometrics_capabilities"
    assert "Create Flare reading" in result[0]
    assert "flare readings page" in result[0]
    assert "Create LACT readings" not in result[0]


def test_show_me_operational_view_extracts_search_text_before_month():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="Show me well tests for 4048 in June 2026",
        today=date(2026, 7, 3),
    )

    assert result is not None
    assert calls[0][1] == {
        "view_type": "well_tests",
        "start_date": "2026-06-01",
        "end_date": "2026-06-30",
        "search_text": "4048",
        "status": None,
    }


def test_show_me_resolves_named_start_date_until_today():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="show me oil production from june 1st until today",
        today=date(2026, 7, 4),
    )

    assert result is not None
    assert calls[0][1]["start_date"] == "2026-06-01"
    assert calls[0][1]["end_date"] == "2026-07-04"
    assert result[2]["model_calls"] == 0


def test_show_me_resolves_explicit_range_connector_formats():
    examples = (
        ("from 2026-06-01 through 2026-07-04", "2026-06-01", "2026-07-04"),
        ("from 06/01/2026 to 07/04/2026", "2026-06-01", "2026-07-04"),
        ("between june 1st and july 4th", "2026-06-01", "2026-07-04"),
        ("from june until today", "2026-06-01", "2026-07-04"),
    )

    for date_text, expected_start, expected_end in examples:
        calls = []
        result = try_answer_navigation_request(
            tools=make_tools(calls),
            question=f"show me oil production {date_text}",
            today=date(2026, 7, 4),
        )

        assert result is not None
        assert calls[0][1]["start_date"] == expected_start
        assert calls[0][1]["end_date"] == expected_end


def test_show_me_separates_trailing_search_text_from_explicit_date_range():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question=(
            "show me tank readings from June 2nd until June 10th "
            "for 15-7 Water"
        ),
        today=date(2026, 7, 4),
    )

    assert result is not None
    assert calls[0][1]["view_type"] == "tank_readings"
    assert calls[0][1]["start_date"] == "2026-06-02"
    assert calls[0][1]["end_date"] == "2026-06-10"
    assert calls[0][1]["search_text"] == "15-7 water"
    assert result[2]["model_calls"] == 0


def test_explicit_range_connectors_allow_trailing_search_text():
    examples = (
        "from 2026-06-02 through 2026-06-10 for 15-7 Water",
        "from 06/02/2026 to 06/10/2026 for 15-7 Water",
        "between June 2nd and June 10th for 15-7 Water",
    )

    for date_text in examples:
        calls = []
        result = try_answer_navigation_request(
            tools=make_tools(calls),
            question=f"show me tank readings {date_text}",
            today=date(2026, 7, 4),
        )

        assert result is not None
        assert calls[0][1]["start_date"] == "2026-06-02"
        assert calls[0][1]["end_date"] == "2026-06-10"
        assert calls[0][1]["search_text"] == "15-7 water"


def test_unrecognized_show_me_request_falls_through():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="Show me everything interesting",
        today=date(2026, 7, 3),
    )

    assert result is None
    assert calls == []


def test_data_entry_routes_entity_and_labelled_values_without_model():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question=(
            "Create a flare reading for Battery 6 Flare with pressure 10 "
            "and volume 20 today"
        ),
        today=date(2026, 7, 3),
    )

    assert result is not None
    arguments = calls[0][1]
    assert arguments["entry_type"] == "flare_reading"
    assert arguments["entity_name"] == "battery 6 flare"
    assert arguments["values"] == {
        "date": "2026-07-03",
        "pressure": 10.0,
        "volume": 20.0,
    }
    assert result[2]["model_calls"] == 0


def test_tank_reading_voice_fields_are_parsed_without_model():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question=(
            "Create a tank reading for tank 5–2 float over top level fit five "
            "top level inch two water level feet two water level inch one"
        ),
        today=date(2026, 7, 7),
    )

    assert result is not None
    arguments = calls[0][1]
    assert arguments["entry_type"] == "tank_reading"
    assert arguments["entity_name"] == "tank 5–2 float over"
    assert arguments["values"] == {
        "top_level_feet": 5.0,
        "top_level_inches": 2.0,
        "water_level_feet": 2.0,
        "water_level_inches": 1.0,
    }
    assert result[2]["model_calls"] == 0


def test_tank_reading_digit_fields_are_parsed_without_model():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question=(
            "Create a tank reading for Tank 5-2 Float Over top level feet 5 "
            "top level inches 2 water level feet 2 water level inches 1"
        ),
        today=date(2026, 7, 7),
    )

    assert result is not None
    assert calls[0][1]["entity_name"] == "tank 5-2 float over"
    assert calls[0][1]["values"] == {
        "top_level_feet": 5.0,
        "top_level_inches": 2.0,
        "water_level_feet": 2.0,
        "water_level_inches": 1.0,
    }


def test_generic_tank_reading_voice_fields_are_parsed_without_model():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question=(
            "Create a reading for tank 5-2 float over top level feet five "
            "water level inch one"
        ),
        today=date(2026, 7, 7),
    )

    assert result is not None
    assert calls[0][1]["entry_type"] == "tank_reading"
    assert calls[0][1]["entity_name"] == "tank 5-2 float over"
    assert calls[0][1]["values"] == {
        "top_level_feet": 5.0,
        "water_level_inches": 1.0,
    }


def test_generic_reading_without_tank_marker_uses_tank_fields_as_signal():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question=(
            "Create a reading for 5–2 float over top level fit five "
            "top level inch two water level feet two water level inch one"
        ),
        today=date(2026, 7, 7),
    )

    assert result is not None
    assert calls[0][1]["entry_type"] == "tank_reading"
    assert calls[0][1]["entity_name"] == "5–2 float over"
    assert calls[0][1]["values"] == {
        "top_level_feet": 5.0,
        "top_level_inches": 2.0,
        "water_level_feet": 2.0,
        "water_level_inches": 1.0,
    }


@pytest.mark.parametrize(
    ("question", "entry_type", "entity_name", "values"),
    [
        (
            "create reading for Battery 2 Water Transfer flow 20 total 25",
            "flow_meter_reading",
            "battery 2 water transfer",
            {"total": 25.0, "flow": 20.0},
        ),
        (
            "create reading for Injection 1 suction pressure 10 discharge pressure 20",
            "pump_reading",
            "injection 1",
            {"suction_pressure": 10.0, "discharge_pressure": 20.0},
        ),
        (
            "create reading for Plant 1 flow rate 10 suction pressure 20 discharge pressure 30",
            "water_plant_reading",
            "plant 1",
            {"flow_rate": 10.0, "suction_pressure": 20.0, "discharge_pressure": 30.0},
        ),
        (
            "create reading for Battery 6 with pressure 10 volume 20",
            "flare_reading",
            "battery 6",
            {"pressure": 10.0, "volume": 20.0},
        ),
        (
            "create reading for Battery 6 with oil intake 10 pressure 20 temperature 90",
            "treater_reading",
            "battery 6",
            {"oil_intake": 10.0, "pressure": 20.0, "temperature": 90.0},
        ),
        (
            "create reading for Separator 1 inlet 10 oil off 20",
            "knockout_reading",
            "separator 1",
            {"inlet": 10.0, "oil_off": 20.0},
        ),
        (
            "create reading for Sales 1 reading 4545 temperature 80 b s w one",
            "lact_reading",
            "sales 1",
            {"reading": 4545.0, "temperature": 80.0, "bs_w": 1.0},
        ),
        (
            "create reading for 4048 flow rate 10 total 20 tbg 30 csg 40",
            "well_injection",
            "4048",
            {"flow_rate": 10.0, "total": 20.0, "tbg": 30.0, "csg": 40.0},
        ),
        (
            "create reading for 4048 level 300",
            "well_fluid",
            "4048",
            {"level": 300.0},
        ),
    ],
)
def test_generic_reading_infers_type_from_field_signature_without_equipment_marker(
    question,
    entry_type,
    entity_name,
    values,
):
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question=question,
        today=date(2026, 7, 7),
    )

    assert result is not None
    assert calls[0][1]["entry_type"] == entry_type
    assert calls[0][1]["entity_name"] == entity_name
    assert calls[0][1]["values"] == values
    assert result[2]["model_calls"] == 0


@pytest.mark.parametrize(
    ("question", "entry_type", "entity_name", "values"),
    [
        (
            "Create a reading for Battery 6 Flare with pressure ten volume twenty",
            "flare_reading",
            "battery 6 flare",
            {"pressure": 10.0, "volume": 20.0},
        ),
        (
            "Create a reading for Battery 6 Flow Meter with total one flow two odometer three",
            "flow_meter_reading",
            "battery 6 flow meter",
            {"total": 1.0, "flow": 2.0, "odometer": 3.0},
        ),
        (
            "Create a reading for Injection Pump 1 with suction pressure 10 discharge pressure 20",
            "pump_reading",
            "injection pump 1",
            {"suction_pressure": 10.0, "discharge_pressure": 20.0},
        ),
        (
            "Create a reading for Battery 6 Treater with oil intake 10 pressure 20 temperature 90",
            "treater_reading",
            "battery 6 treater",
            {"oil_intake": 10.0, "pressure": 20.0, "temperature": 90.0},
        ),
        (
            "Create a reading for Battery 6 Knock Out with inlet 10 oil off 20",
            "knockout_reading",
            "battery 6 knock out",
            {"inlet": 10.0, "oil_off": 20.0},
        ),
        (
            "Create a reading for Water Plant 1 with flow rate 10 suction pressure 20 discharge pressure 30",
            "water_plant_reading",
            "water plant 1",
            {"flow_rate": 10.0, "suction_pressure": 20.0, "discharge_pressure": 30.0},
        ),
        (
            "Create a reading for Battery 6 Lact with reading 4545 temperature 80 b s w one",
            "lact_reading",
            "battery 6 lact",
            {"reading": 4545.0, "temperature": 80.0, "bs_w": 1.0},
        ),
        (
            "Create a well test for 4048 with oil 10 water 20 gas 30 pip 40 m temp 90 amps 15 tbgp 100 csgp 200 fluid level 300 runtime 24",
            "well_test",
            "4048",
            {
                "oil": 10.0,
                "water": 20.0,
                "gas": 30.0,
                "pip": 40.0,
                "m_temp": 90.0,
                "amps": 15.0,
                "tbgp": 100.0,
                "csgp": 200.0,
                "fluid_level": 300.0,
                "runtime": 24.0,
            },
        ),
        (
            "create well test for 4048 oil 3 water 20 gas 6 run time 24",
            "well_test",
            "4048",
            {
                "oil": 3.0,
                "water": 20.0,
                "gas": 6.0,
                "runtime": 24.0,
            },
        ),
        (
            "Create a well fluid level for 4048 with level 300",
            "well_fluid",
            "4048",
            {"level": 300.0},
        ),
        (
            "Create a well injection for 4048 with flow rate 10 total 20 tbg 30 csg 40",
            "well_injection",
            "4048",
            {"flow_rate": 10.0, "total": 20.0, "tbg": 30.0, "csg": 40.0},
        ),
        (
            "Create a well shutdown for 4048",
            "well_shutdown",
            "4048",
            {},
        ),
        (
            "Create a water draw for 5-2 Float Over with initial feet five initial inch two final feet four final inch one",
            "water_draw",
            "5-2 float over",
            {
                "initial_feet": 5.0,
                "initial_inches": 2.0,
                "final_feet": 4.0,
                "final_inches": 1.0,
            },
        ),
        (
            "Create a run ticket for 5-2 Float Over with number 123 obs grav 40 obs temp 80 b s w one corr grav 42 gov 100 nsv 99 initial feet five initial qtr two final feet four final qtr one",
            "run_ticket",
            "5-2 float over",
            {
                "number": 123.0,
                "obs_grav": 40.0,
                "obs_temp": 80.0,
                "b_s_w": 1.0,
                "corr_grav": 42.0,
                "gov": 100.0,
                "nsv": 99.0,
                "initial_feet": 5.0,
                "initial_qtr": 2.0,
                "final_feet": 4.0,
                "final_qtr": 1.0,
            },
        ),
    ],
)
def test_data_entry_supported_entity_types_pass_labelled_values_without_model(
    question,
    entry_type,
    entity_name,
    values,
):
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question=question,
        today=date(2026, 7, 7),
    )

    assert result is not None
    assert calls[0][1]["entry_type"] == entry_type
    assert calls[0][1]["entity_name"] == entity_name
    assert calls[0][1]["values"] == values
    assert result[2]["model_calls"] == 0


def test_generic_reading_and_client_clarification_are_deterministic():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls, entry_status="needs_clarification"),
        question="Record a reading",
        today=date(2026, 7, 3),
    )

    assert result is not None
    assert calls[0][1]["entry_type"] == "generic_reading"
    assert "LACT - LACT 1" in result[0]


def test_generic_tank_reading_infers_water_tank_and_level_fields():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="create a reading for 2-4 water feet 10 inches 10",
        today=date(2026, 7, 9),
    )

    assert result is not None
    assert calls[0][1]["entry_type"] == "tank_reading"
    assert calls[0][1]["entity_name"] == "2-4 water"
    assert calls[0][1]["values"] == {"feet": 10.0, "inches": 10.0}
    assert result[2]["model_calls"] == 0


@pytest.mark.parametrize(
    ("question", "expected_values"),
    [
        (
            "create a shutdown for 6344 start yesterday 14:30",
            {"start": "2026-07-27 14:30"},
        ),
        (
            "create a shutdown for 6344 start yesterday 14:30 reason paraffin",
            {"start": "2026-07-27 14:30", "downtime_code": "PRF"},
        ),
        (
            "create a shutdown for 6344 start yesterday 14:30 downtime reason PRF",
            {"start": "2026-07-27 14:30", "downtime_code": "PRF"},
        ),
        (
            "create a shutdown for 6344 start yesterday 14:30 end today 09:00",
            {
                "start": "2026-07-27 14:30",
                "end": "2026-07-28 09:00",
            },
        ),
        (
            "create a shutdown for 6344 starting yesterday at 2:30 pm due to paraffin",
            {"start": "2026-07-27 14:30", "downtime_code": "PRF"},
        ),
    ],
)
def test_shutdown_create_prefill_is_deterministic(question, expected_values):
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question=question,
        today=date(2026, 7, 28),
    )

    assert result is not None
    assert calls[0][0] == "prepare_data_entry"
    assert calls[0][1]["entry_type"] == "well_shutdown"
    assert calls[0][1]["entity_name"] == "6344"
    assert calls[0][1]["values"] == expected_values
    assert result[2]["model_calls"] == 0


def test_shutdown_create_with_unknown_reason_does_not_guess_code():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="create a shutdown for 6344 start yesterday 14:30 reason pumped off",
        today=date(2026, 7, 28),
    )

    assert result is not None
    assert calls[0][1]["values"] == {"start": "2026-07-27 14:30"}


def test_shutdown_create_with_unsupported_hours_falls_through_to_model():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="Create a well shutdown for 4048 with hours 6",
        today=date(2026, 7, 3),
    )

    assert result is None
    assert calls == []


def test_unknown_numeric_entry_value_falls_through_to_model():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="Create a well test for 4048 with unknown metric 25",
        today=date(2026, 7, 3),
    )

    assert result is None
    assert calls == []


def test_multiple_entry_types_fall_through_to_model():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="Create LACT and flare readings",
        today=date(2026, 7, 3),
    )

    assert result is None
    assert calls == []


def test_generic_reading_infers_lact_type_from_complete_entity_phrase():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="create a reading for Battery 6 Lact with reading 4545",
        today=date(2026, 7, 4),
    )

    assert result is not None
    assert calls[0][1]["entry_type"] == "lact_reading"
    assert calls[0][1]["entity_name"] == "battery 6 lact"
    assert calls[0][1]["values"] == {"reading": 4545.0}
    assert result[2]["model_calls"] == 0


def test_generic_reading_infers_every_supported_equipment_marker():
    examples = {
        "Battery 6 Flare": "flare_reading",
        "Battery 6 Flow Meter": "flow_meter_reading",
        "Injection Pump 1": "pump_reading",
        "Battery 6 Treater": "treater_reading",
        "Battery 6 Knock Out": "knockout_reading",
        "Water Plant 1": "water_plant_reading",
        "Battery 6 Lact": "lact_reading",
        "Tank 6-1": "tank_reading",
    }

    for entity_name, expected_type in examples.items():
        calls = []
        result = try_answer_navigation_request(
            tools=make_tools(calls),
            question=f"Create a reading for {entity_name}",
            today=date(2026, 7, 4),
        )

        assert result is not None
        assert calls[0][1]["entry_type"] == expected_type
        assert calls[0][1]["entity_name"] == entity_name.lower()


def test_generic_reading_with_multiple_entity_markers_falls_through():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="Create a reading for Battery 6 Lact Flare",
        today=date(2026, 7, 4),
    )

    assert result is None
    assert calls == []


def test_general_note_extracts_unquoted_trailing_comments():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="create general notes comments I'm having a good day",
        today=date(2026, 7, 4),
    )

    assert result is not None
    assert calls[0][1]["entry_type"] == "general_note"
    assert calls[0][1]["values"] == {
        "comments": "i'm having a good day",
    }
    assert result[2]["model_calls"] == 0


def test_unquoted_comments_can_contain_numbers_without_llm_fallback():
    calls = []
    result = try_answer_navigation_request(
        tools=make_tools(calls),
        question="create general note comments checked 3 devices",
        today=date(2026, 7, 4),
    )

    assert result is not None
    assert calls[0][1]["values"]["comments"] == "checked 3 devices"
