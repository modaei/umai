import json

from umai.clients.rod_pump_analysis_client import RodPumpAnalysisError
from umai.tools.rod_pump_analysis_tools import build_rod_pump_analysis_tools


class Client:
    def analyze(self, site_id, well_name, start_time, end_time):
        return {"site_id": site_id, "well": {"name": well_name}, "diagnoses": []}

    def rank_wells(self, site_id, as_of_time):
        return {"site_id": site_id, "wells": []}


class FailingClient(Client):
    def analyze(self, *args):
        raise RodPumpAnalysisError("not available")


def tool(tools, name):
    return next(item for item in tools if item.name == name)


def test_analysis_tool_scopes_request_to_site():
    result = json.loads(tool(build_rod_pump_analysis_tools(Client(), 4), "analyze_rod_pump").invoke({"well_name": "5823"}))
    assert result["ok"] is True
    assert result["site_id"] == 4


def test_analysis_tool_returns_controlled_error():
    result = json.loads(tool(build_rod_pump_analysis_tools(FailingClient(), 4), "analyze_rod_pump").invoke({"well_name": "5823"}))
    assert result == {"ok": False, "error": "not available"}


def test_fleet_ranking_is_not_exposed_as_a_chat_tool():
    names = [item.name for item in build_rod_pump_analysis_tools(Client(), 4)]
    assert names == ["analyze_rod_pump"]
