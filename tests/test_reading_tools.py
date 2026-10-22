import json

from umai.tools.reading_tools import build_reading_tools


class FakeReadingClient:
    def __init__(self):
        self.range_calls = []

    def list_reading_types(self):
        return []

    def all_missing_readings_for_date(self, site_id, reading_date):
        return {
            "reading_type": "all_supported_missing_readings",
            "date": reading_date,
            "missing_count": 0,
            "groups": [],
        }

    def all_missing_readings_for_range(self, site_id, start_date, end_date):
        self.range_calls.append(
            {"site_id": site_id, "start_date": start_date, "end_date": end_date}
        )
        return {
            "reading_type": "all_supported_missing_readings",
            "start_date": start_date,
            "end_date": end_date,
            "day_count": 7,
            "missing_count": 0,
            "dates": [],
        }

    def search_readings(
        self,
        site_id,
        reading_type,
        start_date,
        end_date,
        entity_name=None,
        filters=None,
    ):
        return {
            "reading_type": reading_type,
            "start_date": start_date,
            "end_date": end_date,
            "rows": [{"entity_name": entity_name, "reading": 42}],
        }

    def search_tank_readings(
        self,
        site_id,
        start_date,
        end_date,
        battery_name=None,
        tank_name=None,
        tank_key=None,
        tank_type=None,
        contains=None,
        monitored=None,
        disable_reading=None,
        filters=None,
    ):
        return {
            "reading_type": "tank",
            "start_date": start_date,
            "end_date": end_date,
            "filters": {
                "battery_name": battery_name,
                "contains": contains,
                "computed_filters": filters or [],
            },
            "count": 1,
            "readings": [
                {
                    "tank_name": "Mixed A",
                    "battery_name": battery_name,
                    "oil_volume": 10,
                }
            ],
        }

    def search_equipment_readings(
        self,
        site_id,
        reading_type,
        start_date,
        end_date,
        battery_name=None,
        entity_name=None,
        entity_key=None,
        monitored=None,
        disable_reading=None,
        equipment_filters=None,
        reading_filters=None,
        computed_filters=None,
        contains=None,
    ):
        return {
            "reading_type": reading_type,
            "start_date": start_date,
            "end_date": end_date,
            "filters": {
                "battery_name": battery_name,
                "entity_name": entity_name,
                "equipment_filters": equipment_filters or [],
                "reading_filters": reading_filters or [],
                "computed_filters": computed_filters or [],
                "contains": contains,
            },
            "count": 1,
            "readings": [
                {
                    "entity_name": "FM Battery 6 Gas",
                    "battery_name": battery_name,
                    "total": 150,
                }
            ],
        }

    def list_equipment(
        self,
        site_id,
        equipment_type,
        battery_name=None,
        entity_name=None,
        entity_key=None,
        monitored=None,
        disable_reading=None,
        equipment_filters=None,
    ):
        return {
            "equipment_type": equipment_type,
            "filters": {
                "battery_name": battery_name,
                "equipment_filters": equipment_filters or [],
            },
            "count": 1,
            "entities": [
                {
                    "entity_name": "Tank 6-1 Oil",
                    "battery_name": battery_name,
                }
            ],
        }

    def get_reading_for_entity(
        self,
        site_id,
        entity_name,
        reading_date,
        reading_type=None,
    ):
        return {
            "reading_type": reading_type or "flow_meter",
            "date": reading_date,
            "entity": {
                "reading_type": reading_type or "flow_meter",
                "entity_display_name": f"Flow Meter - {entity_name}",
            },
            "count": 1,
            "readings": [{"entity_display_name": f"Flow Meter - {entity_name}", "total": 0}],
        }


class FakeOperationalContextStore:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "query": kwargs["query"],
            "count": 1,
            "matches": [{"source_type": "general_note", "text": "Reading context"}],
        }


def tool_by_name(tools, name):
    return next(tool for tool in tools if tool.name == name)


def test_find_all_missing_readings_tool_uses_single_client_call():
    tools = build_reading_tools(FakeReadingClient(), 4)
    result = json.loads(
        tool_by_name(tools, "find_all_missing_readings").invoke(
            {"reading_date": "2026-05-20"}
        )
    )

    assert result["ok"] is True
    assert result["reading_type"] == "all_supported_missing_readings"
    assert result["date"] == "2026-05-20"


def test_find_all_missing_readings_for_range_tool_uses_single_client_call():
    client = FakeReadingClient()
    tools = build_reading_tools(client, 4)
    result = json.loads(
        tool_by_name(tools, "find_all_missing_readings_for_range").invoke(
            {"start_date": "2026-06-15", "end_date": "2026-06-21"}
        )
    )

    assert result["ok"] is True
    assert result["reading_type"] == "all_supported_missing_readings"
    assert result["start_date"] == "2026-06-15"
    assert result["end_date"] == "2026-06-21"
    assert client.range_calls == [
        {
            "site_id": 4,
            "start_date": "2026-06-15",
            "end_date": "2026-06-21",
        }
    ]


def test_search_readings_tool_does_not_add_operational_context():
    store = FakeOperationalContextStore()
    tools = build_reading_tools(
        FakeReadingClient(),
        4,
        operational_context_store=store,
    )
    result = json.loads(
        tool_by_name(tools, "search_readings").invoke(
            {
                "reading_type": "lact",
                "start_date": "2026-05-01",
                "end_date": "2026-05-31",
                "entity_name": "LACT 1",
            }
        )
    )

    assert result["ok"] is True
    assert "operational_context" not in result
    assert store.calls == []


def test_search_tank_readings_tool_uses_battery_and_computed_filters():
    tools = build_reading_tools(FakeReadingClient(), 4)
    result = json.loads(
        tool_by_name(tools, "search_tank_readings").invoke(
            {
                "start_date": "2026-06-24",
                "end_date": "2026-06-24",
                "battery_name": "Battery 6",
                "contains": "oil",
                "filters": [
                    {"field": "oil_volume", "operator": ">", "value": 0},
                ],
            }
        )
    )

    assert result["ok"] is True
    assert result["filters"]["battery_name"] == "Battery 6"
    assert result["filters"]["contains"] == "oil"
    assert result["readings"][0]["oil_volume"] == 10


def test_search_tank_readings_tool_describes_content_aware_oil_filters():
    tool = tool_by_name(build_reading_tools(FakeReadingClient(), 4), "search_tank_readings")

    assert "only oil and water-oil contents are oil-capable" in tool.description
    assert "contains=oil" in tool.description
    assert "recoverable_oil_volume > 0" in tool.description

    contains_description = tool.args_schema.model_fields["contains"].description
    assert "persisted tanks.contents" in contains_description
    assert "positive gross oil_volume" in contains_description


def test_search_equipment_readings_tool_uses_relation_and_filters():
    tools = build_reading_tools(FakeReadingClient(), 4)
    result = json.loads(
        tool_by_name(tools, "search_equipment_readings").invoke(
            {
                "reading_type": "flow_meter",
                "start_date": "2026-06-24",
                "end_date": "2026-06-24",
                "battery_name": "Battery 6",
                "equipment_filters": [
                    {"field": "type", "operator": "=", "value": "gas"},
                ],
                "reading_filters": [
                    {"field": "total", "operator": ">", "value": 100},
                ],
            }
        )
    )

    assert result["ok"] is True
    assert result["filters"]["battery_name"] == "Battery 6"
    assert result["filters"]["equipment_filters"] == [
        {"field": "type", "operator": "=", "value": "gas"}
    ]
    assert result["filters"]["reading_filters"] == [
        {"field": "total", "operator": ">", "value": 100.0}
    ]
    assert result["readings"][0]["total"] == 150


def test_list_equipment_tool_does_not_require_reading_date():
    tools = build_reading_tools(FakeReadingClient(), 4)
    result = json.loads(
        tool_by_name(tools, "list_equipment").invoke(
            {
                "equipment_type": "tank",
                "battery_name": "Battery 6",
            }
        )
    )

    assert result["ok"] is True
    assert result["equipment_type"] == "tank"
    assert result["filters"]["battery_name"] == "Battery 6"
    assert result["entities"][0]["entity_name"] == "Tank 6-1 Oil"


def test_get_reading_for_entity_tool_uses_resolver_client_call():
    tools = build_reading_tools(FakeReadingClient(), 4)
    result = json.loads(
        tool_by_name(tools, "get_reading_for_entity").invoke(
            {
                "entity_name": "Battery 2 Vent",
                "reading_date": "2026-06-01",
            }
        )
    )

    assert result["ok"] is True
    assert result["reading_type"] == "flow_meter"
    assert result["entity"]["entity_display_name"] == "Flow Meter - Battery 2 Vent"
