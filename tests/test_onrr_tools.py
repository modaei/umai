import json

from umai.tools.onrr_tools import build_onrr_tools


class FakeOnrrClient:
    def list_onrr_codes(self):
        return {"count": 1, "onrr_codes": [{"code": "POW"}]}

    def explain_onrr_code(self, code):
        return {
            "code": code,
            "found": True,
            "onrr_code": {
                "code": "POW",
                "active_well": True,
                "injection_well": False,
                "description": "Producing oil well",
            },
        }

    def get_well_onrr_status_as_of_date(self, site_id, well_name, as_of_date):
        return {
            "site_id": site_id,
            "well_name": well_name,
            "date": as_of_date,
            "found": True,
            "status": {"well": "Well - HARTZOG DRAW UNIT 4048", "onrr_code": "POW"},
        }

    def count_wells_by_onrr_status(self, site_id, as_of_date, status):
        return {
            "site_id": site_id,
            "date": as_of_date,
            "status": status,
            "counts": {
                "active": 2,
                "producing": 1,
                "injection": 1,
                "inactive": 0,
            },
        }


def tool_by_name(tools, name):
    return next(tool for tool in tools if tool.name == name)


def test_explain_onrr_code_tool_returns_description():
    tools = build_onrr_tools(FakeOnrrClient(), site_id=4)

    result = json.loads(
        tool_by_name(tools, "explain_onrr_code").invoke({"code": "POW"})
    )

    assert result["ok"] is True
    assert result["onrr_code"]["description"] == "Producing oil well"


def test_well_onrr_status_tool_passes_site_id():
    tools = build_onrr_tools(FakeOnrrClient(), site_id=4)

    result = json.loads(
        tool_by_name(tools, "get_well_onrr_status_as_of_date").invoke(
            {"well_name": "4048", "as_of_date": "2026-06-20"}
        )
    )

    assert result["ok"] is True
    assert result["site_id"] == 4
    assert result["status"]["onrr_code"] == "POW"


def test_count_wells_by_onrr_status_tool_returns_counts():
    tools = build_onrr_tools(FakeOnrrClient(), site_id=4)

    result = json.loads(
        tool_by_name(tools, "count_wells_by_onrr_status").invoke(
            {"as_of_date": "2026-06-20", "status": "producing"}
        )
    )

    assert result["ok"] is True
    assert result["counts"]["producing"] == 1
