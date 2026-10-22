import json

from umai.tools.shutdown_tools import build_shutdown_tools


class FakeShutdownClient:
    def summarize_shutdown_causes(self, site_id, start_date, end_date):
        return {
            "site_id": site_id,
            "start_date": start_date,
            "end_date": end_date,
            "main_cause": {"downtime_reason": "Paraffin"},
            "causes": [],
        }

    def get_shutdowns(self, site_id, start_date, end_date):
        return {
            "site_id": site_id,
            "start_date": start_date,
            "end_date": end_date,
            "shutdowns": [],
        }

    def get_current_shutdowns(self, site_id, as_of_date=None):
        return {
            "site_id": site_id,
            "as_of_date": as_of_date or "2026-06-24",
            "shutdowns": [],
        }

    def get_active_wells(self, site_id, active_date):
        return {
            "site_id": site_id,
            "date": active_date,
            "active_count": 2,
            "inactive_count": 1,
            "active_wells": [{"well": "Well - 11-1-1 Oil"}],
            "inactive_wells": [{"well": "Well - 12-2-1 Oil"}],
        }

    def get_producing_wells(self, site_id, producing_date, filters=None):
        return {
            "site_id": site_id,
            "date": producing_date,
            "filters": filters or [],
            "producing_count": 1,
            "non_producing_count": 1,
            "partial_shutdown_count": 0,
            "producing_wells": [{"well": "Well - 11-1-1 Oil"}],
            "non_producing_wells": [
                {
                    "well": "Well - 12-2-1 Injection",
                    "status": "onrr_injection_well",
                    "onrr_code": "INJ",
                    "onrr_code_description": "Active injection well",
                }
            ],
            "partial_shutdown_wells": [],
            "partial_shutdown_well_names": [],
            "partial_shutdown_summary": None,
        }

    def get_producing_wells_for_range(
        self, site_id, start_date, end_date, range_mode="any_day", filters=None
    ):
        return {
            "site_id": site_id,
            "start_date": start_date,
            "end_date": end_date,
            "range_mode": range_mode,
            "filters": filters or [],
            "producing_count": 2,
            "producing_wells": [
                {"well": "Well - 11-1-1 Oil"},
                {"well": "Well - 13-1-1 Oil"},
            ],
        }

    def list_downtime_codes(self):
        return {"downtime_codes": {}}


class FakeOperationalContextStore:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "query": kwargs["query"],
            "count": 1,
            "matches": [{"source_type": "chart_note", "text": "Shutdown context"}],
        }


def tool_by_name(tools, name):
    return next(tool for tool in tools if tool.name == name)


def test_summarize_shutdown_causes_tool_adds_operational_context():
    store = FakeOperationalContextStore()
    tools = build_shutdown_tools(
        FakeShutdownClient(),
        site_id=4,
        operational_context_store=store,
    )

    result = json.loads(
        tool_by_name(tools, "summarize_shutdown_causes").invoke(
            {
                "start_date": "2026-05-01",
                "end_date": "2026-05-31",
            }
        )
    )

    assert result["ok"] is True
    assert result["operational_context"]["matches"][0]["text"] == "Shutdown context"
    assert store.calls[0]["site_id"] == 4
    assert "Paraffin" in store.calls[0]["query"]


def test_get_active_wells_tool_returns_counts():
    tools = build_shutdown_tools(FakeShutdownClient(), site_id=4)

    result = json.loads(
        tool_by_name(tools, "get_active_wells").invoke(
            {"active_date": "2026-06-21"}
        )
    )

    assert result["ok"] is True
    assert result["active_count"] == 2
    assert result["inactive_count"] == 1
    assert "partial_shutdown_count" not in result


def test_get_producing_wells_tool_returns_counts():
    tools = build_shutdown_tools(FakeShutdownClient(), site_id=4)

    result = json.loads(
        tool_by_name(tools, "get_producing_wells").invoke(
            {"producing_date": "2026-06-21"}
        )
    )

    assert result["ok"] is True
    assert result["producing_count"] == 1
    assert result["non_producing_wells"][0]["status"] == "onrr_injection_well"


def test_get_producing_wells_tool_supports_any_day_range():
    tools = build_shutdown_tools(FakeShutdownClient(), site_id=4)

    result = json.loads(
        tool_by_name(tools, "get_producing_wells").invoke(
            {
                "start_date": "2026-06-01",
                "end_date": "2026-06-30",
                "range_mode": "any_day",
            }
        )
    )

    assert result["ok"] is True
    assert result["producing_count"] == 2
    assert result["range_mode"] == "any_day"


def test_get_producing_wells_tool_accepts_filters():
    tools = build_shutdown_tools(FakeShutdownClient(), site_id=4)

    result = json.loads(
        tool_by_name(tools, "get_producing_wells").invoke(
            {
                "producing_date": "2026-06-21",
                "filters": [{"field": "battery", "value": "Battery 6"}],
            }
        )
    )

    assert result["ok"] is True
    assert result["filters"] == [{"field": "battery", "value": "Battery 6"}]
