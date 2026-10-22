from pathlib import Path

from umai.clients.capability_client import CapabilityClient
from umai.tools.capability_tools import build_capability_tools


def make_client() -> CapabilityClient:
    return CapabilityClient(Path("knowledge/capabilities"))


def test_search_finds_well_shutdown_guidance():
    result = make_client().search("A well is down. How can I register that?")

    assert result["count"] >= 1
    assert result["matches"][0]["capability"] == "Capability: Well Shutdowns"
    assert "shutdowns" in result["matches"][0]["summary"]


def test_search_finds_production_allocation_guidance():
    result = make_client().search(
        "How can I know how much well 6243 contributed to oil production?"
    )

    assert result["count"] >= 1
    assert result["matches"][0]["capability"] == (
        "Capability: Production Allocation Report"
    )
    summary = result["matches"][0]["summary"].lower()
    assert "production allocation report" in summary
    assert "do not invent ui steps" in summary
    assert "only point the user" in summary


def test_search_finds_lact_reading_create_guidance():
    result = make_client().search("How to register the value that I read for a lact?")

    assert result["count"] >= 1
    assert result["matches"][0]["capability"] == "Capability: Raw Field Readings"
    summary = result["matches"][0]["summary"].lower()
    assert "create a new lact reading" in summary
    assert "do not include" in summary


def test_search_finds_alarm_page_guidance_for_device_error_state():
    result = make_client().search("How to know if a device is in error state?")

    assert result["count"] >= 1
    assert result["matches"][0]["capability"] == "Capability: Alarms"
    summary = result["matches"][0]["summary"].lower()
    assert "alarms page" in summary
    assert "do not tell the user to use well timelines" in summary


def test_search_finds_water_injection_report_guidance():
    result = make_client().search("How to know how much water was injected last week?")

    assert result["count"] >= 1
    assert result["matches"][0]["capability"] == "Capability: Reports"
    summary = result["matches"][0]["summary"].lower()
    assert "water injection report" in summary
    assert "do not run the report" in summary
    assert "ask whether the user wants the assistant to run" in summary


def test_capability_tool_description_excludes_data_retrieval():
    tool = build_capability_tools(make_client())[0]

    assert "explicitly asks for software help" in tool.description
    assert "Do not use this tool to retrieve, compare, calculate, rank, or summarize" in tool.description
    assert "actual operational values or date-range data" in tool.description
