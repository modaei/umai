import json

from umai.clients.rod_pump_health_client import RodPumpHealthError
from umai.tools.rod_pump_health_tools import build_rod_pump_health_tools


WELL = {"id": 295, "name": "HARTZOG DRAW UNIT 3051", "key": "HDU_3051"}


class Client:
    def __init__(self, in_progress=False):
        self.in_progress = in_progress
        self.calls = []

    def well_health(self, site_id, well_name):
        self.calls.append(("well_health", site_id, well_name))
        return {"well": WELL, "available": True, "health_status": "watch",
                "refresh": {"in_progress": self.in_progress}}

    def site_health(self, site_id):
        self.calls.append(("site_health", site_id))
        return {"site_id": site_id, "wells": []}


class FailingClient(Client):
    def well_health(self, *args):
        raise RodPumpHealthError("not available")


def tool(tools, name):
    return next(item for item in tools if item.name == name)


def invoke(client, name, arguments=None):
    return json.loads(tool(build_rod_pump_health_tools(client, 4), name).invoke(arguments or {}))


def test_tools_are_site_bound():
    client = Client()
    assert invoke(client, "get_rod_pump_health", {"well_name": "3051"})["health_status"] == "watch"
    assert invoke(client, "list_rod_pump_health")["site_id"] == 4
    assert client.calls == [("well_health", 4, "3051"), ("site_health", 4)]


def test_tools_return_controlled_errors():
    assert invoke(FailingClient(), "get_rod_pump_health", {"well_name": "3051"}) == {
        "ok": False, "error": "not available",
    }
    assert invoke(FailingClient(), "request_rod_pump_health_rerun", {"well_name": "3051"}) == {
        "ok": False, "error": "not available",
    }


def test_rerun_tool_only_records_the_request():
    result = invoke(Client(), "request_rod_pump_health_rerun", {"well_name": "3051"})

    assert result["ok"] is True
    assert result["status"] == "rerun_requested"
    assert result["well"] == WELL


def test_rerun_tool_refuses_while_a_check_is_running():
    result = invoke(Client(in_progress=True), "request_rod_pump_health_rerun", {"well_name": "3051"})

    assert result["ok"] is False
    assert "already running" in result["error"]
    assert "status" not in result


def test_exposed_tool_names():
    names = [item.name for item in build_rod_pump_health_tools(Client(), 4)]
    assert names == ["get_rod_pump_health", "list_rod_pump_health", "request_rod_pump_health_rerun"]
