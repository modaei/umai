import json

from umai.tools.report_tools import build_report_tools


class FakeReportClient:
    def run_report(self, site_id, report_name, start_date, end_date):
        return [{"date": start_date, "value": 12.5}]


class FakeMonthlyReportClient:
    def __init__(self):
        self.calls = []

    def run_report(self, site_id, report_name, start_date, end_date):
        self.calls.append((site_id, report_name, start_date, end_date))
        return {
            "date_battery_values": {
                "Battery 5": [
                    {"date": 1767225600000, "value": 99},
                ],
                "Battery 6": [
                    {"date": 1767225600000, "value": 10},
                    {"date": 1769904000000, "value": 12.345},
                    {"date": 1772323200000, "value": 7},
                ],
            }
        }


class FakeAllocationReportClient:
    def __init__(self):
        self.calls = []

    def run_report(self, site_id, report_name, start_date, end_date):
        self.calls.append((site_id, report_name, start_date, end_date))
        if report_name == "injection_allocation":
            return [
                {
                    "name": "HARTZOG DRAW UNIT 1001",
                    "onrr_code": "POW",
                    "total_allocated_injection": 12.5,
                },
                {
                    "name": "HARTZOG DRAW UNIT 1002",
                    "onrr_code": "TA",
                    "total_allocated_injection": 7.5,
                },
            ]
        return [
            {
                "name": "HARTZOG DRAW UNIT 1001",
                "onrr_code": "POW",
                "total_oil": 10,
                "total_water": 20,
                "total_gas": 30,
            },
            {
                "name": "HARTZOG DRAW UNIT 1002",
                "onrr_code": "TA",
                "total_oil": 5,
                "total_water": 7,
                "total_gas": 11,
            },
            {
                "name": "HARTZOG DRAW UNIT 2001",
                "onrr_code": "TA",
                "total_oil": 99,
                "total_water": 98,
                "total_gas": 97,
            },
        ]


class FakeWellFilterClient:
    def __init__(self):
        self.calls = []

    def find_wells(self, site_id, well_name=None, filters=None):
        self.calls.append(
            {"site_id": site_id, "well_name": well_name, "filters": filters or []}
        )
        filter_map = {item["field"]: str(item["value"]).lower() for item in filters or []}
        pump = filter_map.get("pump_type", "")
        if pump == "rod":
            wells = [
                {"name": "HARTZOG DRAW UNIT 1001", "pump_type": "ROD"},
                {"name": "HARTZOG DRAW UNIT 1002", "pump_type": "ROD"},
            ]
        elif pump == "esp":
            wells = [{"name": "HARTZOG DRAW UNIT 2001", "pump_type": "ESP"}]
        elif pump == "jet":
            wells = []
        elif pump == "flowing well no lift":
            wells = [{"name": "HARTZOG DRAW UNIT 3001", "pump_type": "flowing well no lift"}]
        else:
            wells = []
        return {"count": len(wells), "wells": wells, "filters": filters or []}


class FakeOperationalContextStore:
    def __init__(self):
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "query": kwargs["query"],
            "count": 1,
            "matches": [{"source_type": "general_note", "text": "Report context"}],
        }


def tool_by_name(tools, name):
    return next(tool for tool in tools if tool.name == name)


def test_run_report_result_exposes_site_name_not_site_id():
    tools = build_report_tools(FakeReportClient(), 4, "HARTZOG DRAW")
    result = json.loads(
        tool_by_name(tools, "run_report").invoke(
            {
                "report_name": "water_injection",
                "start_date": "2026-05-01",
                "end_date": "2026-05-31",
            }
        )
    )

    assert result["ok"] is True
    assert result["site_name"] == "HARTZOG DRAW"
    assert "site_id" not in result
    assert result["report_name"] == "water_injection"


def test_run_report_does_not_add_operational_context():
    store = FakeOperationalContextStore()
    tools = build_report_tools(
        FakeReportClient(),
        4,
        "HARTZOG DRAW",
        operational_context_store=store,
    )
    result = json.loads(
        tool_by_name(tools, "run_report").invoke(
            {
                "report_name": "oil_production",
                "start_date": "2026-05-01",
                "end_date": "2026-05-31",
            }
        )
    )

    assert result["ok"] is True
    assert "operational_context" not in result
    assert store.calls == []


def test_list_available_reports_result_exposes_site_name_not_site_id():
    tools = build_report_tools(FakeReportClient(), 4, "HARTZOG DRAW")
    result = json.loads(tool_by_name(tools, "list_available_reports").invoke({}))

    assert result["site_name"] == "HARTZOG DRAW"
    assert "site_id" not in result
    assert "water_injection" in result["reports"]


def test_summarize_report_by_month_uses_one_annual_report_call():
    client = FakeMonthlyReportClient()
    tools = build_report_tools(client, 4, "HARTZOG DRAW")

    result = json.loads(
        tool_by_name(tools, "summarize_report_by_month").invoke(
            {
                "report_name": "oil_production",
                "year": 2026,
                "battery_name": "6",
            }
        )
    )

    assert client.calls == [(4, "oil_production", "2026-01-01", "2026-12-31")]
    assert result["ok"] is True
    assert result["site_name"] == "HARTZOG DRAW"
    assert result["battery_name"] == "Battery 6"
    assert result["value_key"] == "value"
    assert result["monthly_totals"]["2026-01"] == 10
    assert result["monthly_totals"]["2026-02"] == 12.35
    assert result["monthly_totals"]["2026-03"] == 7
    assert result["monthly_totals"]["2026-04"] == 0
    assert result["matched_daily_rows"] == 3


def test_summarize_well_allocation_filters_rod_wells_case_insensitively():
    report_client = FakeAllocationReportClient()
    well_filter_client = FakeWellFilterClient()
    tools = build_report_tools(
        report_client,
        4,
        "HARTZOG DRAW",
        well_filter_client=well_filter_client,
    )

    result = json.loads(
        tool_by_name(tools, "summarize_well_allocation").invoke(
            {
                "allocation_type": "production",
                "start_date": "2026-06-01",
                "end_date": "2026-06-01",
                "well_filters": [{"field": "pump_type", "value": "rod"}],
            }
        )
    )

    assert report_client.calls == [
        (4, "production_allocation", "2026-06-01", "2026-06-01")
    ]
    assert well_filter_client.calls == [
        {
            "site_id": 4,
            "well_name": None,
            "filters": [{"field": "pump_type", "value": "rod"}],
        }
    ]
    assert result["ok"] is True
    assert result["matched_well_count"] == 2
    assert result["totals"] == {
        "total_oil": 15.0,
        "total_water": 27.0,
        "total_gas": 41.0,
        "total_sold": 0.0,
        "total_vented": 0.0,
        "total_flared": 0.0,
        "total_fuel": 0.0,
    }


def test_summarize_well_allocation_supports_other_pump_types():
    tools = build_report_tools(
        FakeAllocationReportClient(),
        4,
        "HARTZOG DRAW",
        well_filter_client=FakeWellFilterClient(),
    )

    esp = json.loads(
        tool_by_name(tools, "summarize_well_allocation").invoke(
            {
                "allocation_type": "production",
                "start_date": "2026-06-01",
                "end_date": "2026-06-01",
                "well_filters": [{"field": "pump_type", "value": "ESP"}],
            }
        )
    )
    jet = json.loads(
        tool_by_name(tools, "summarize_well_allocation").invoke(
            {
                "allocation_type": "production",
                "start_date": "2026-06-01",
                "end_date": "2026-06-01",
                "well_filters": [{"field": "pump_type", "value": "JET"}],
            }
        )
    )

    assert esp["matched_well_count"] == 1
    assert esp["totals"]["total_oil"] == 99.0
    assert jet["matched_well_count"] == 0
    assert jet["totals"]["total_oil"] == 0.0


def test_summarize_well_allocation_uses_injection_allocation():
    tools = build_report_tools(
        FakeAllocationReportClient(),
        4,
        "HARTZOG DRAW",
        well_filter_client=FakeWellFilterClient(),
    )

    result = json.loads(
        tool_by_name(tools, "summarize_well_allocation").invoke(
            {
                "allocation_type": "injection",
                "start_date": "2026-06-01",
                "end_date": "2026-06-01",
                "well_filters": [{"field": "pump_type", "value": "rod"}],
            }
        )
    )

    assert result["report_name"] == "injection_allocation"
    assert result["totals"] == {"total_allocated_injection": 20.0}


def test_summarize_well_allocation_filters_onrr_code_from_report_rows():
    well_filter_client = FakeWellFilterClient()
    tools = build_report_tools(
        FakeAllocationReportClient(),
        4,
        "HARTZOG DRAW",
        well_filter_client=well_filter_client,
    )

    result = json.loads(
        tool_by_name(tools, "summarize_well_allocation").invoke(
            {
                "allocation_type": "production",
                "start_date": "2026-06-01",
                "end_date": "2026-06-01",
                "well_filters": [{"field": "onrr_code", "value": "TA"}],
            }
        )
    )

    assert well_filter_client.calls == []
    assert result["matched_well_count"] == 2
    assert result["totals"]["total_oil"] == 104.0


def test_summarize_well_allocation_combines_db_and_report_filters():
    tools = build_report_tools(
        FakeAllocationReportClient(),
        4,
        "HARTZOG DRAW",
        well_filter_client=FakeWellFilterClient(),
    )

    result = json.loads(
        tool_by_name(tools, "summarize_well_allocation").invoke(
            {
                "allocation_type": "production",
                "start_date": "2026-06-01",
                "end_date": "2026-06-01",
                "well_filters": [
                    {"field": "pump_type", "value": "rod"},
                    {"field": "onrr_code", "value": "TA"},
                ],
            }
        )
    )

    assert result["matched_well_count"] == 1
    assert result["matched_wells"] == ["HARTZOG DRAW UNIT 1002"]
    assert result["totals"]["total_oil"] == 5.0


def test_list_well_allocation_returns_ranked_per_well_rows():
    tools = build_report_tools(
        FakeAllocationReportClient(),
        4,
        "HARTZOG DRAW",
        well_filter_client=FakeWellFilterClient(),
    )

    result = json.loads(
        tool_by_name(tools, "list_well_allocation").invoke(
            {
                "allocation_type": "production",
                "start_date": "2026-06-01",
                "end_date": "2026-06-05",
                "sort_by": "total_oil",
                "limit": 2,
            }
        )
    )

    assert result["ok"] is True
    assert result["matched_well_count"] == 3
    assert result["sort_by"] == "total_oil"
    assert [row["name"] for row in result["rows"]] == [
        "HARTZOG DRAW UNIT 2001",
        "HARTZOG DRAW UNIT 1001",
    ]
    assert result["rows"][0]["total_oil"] == 99.0
    assert result["rows"][0]["daily_avg_oil"] == 19.8


class FakeFlaggedAllocationReportClient:
    def run_report(self, site_id, report_name, start_date, end_date):
        return [
            {
                "name": "HARTZOG DRAW UNIT 1001",
                "onrr_code": "POW",
                "total_oil": 10,
                "flags": ["Well test up to 90 days old used (limit 60)"],
                "row_type": "well",
            },
            {
                "name": "Unallocated",
                "onrr_code": None,
                "total_oil": 40,
                "flags": ["Oil production on 1 day(s) at Battery 1 had no producing well with a test rate"],
                "row_type": "unallocated",
            },
        ]


def test_allocation_tools_exclude_unallocated_row_and_surface_flags():
    tools = build_report_tools(
        FakeFlaggedAllocationReportClient(),
        4,
        "HARTZOG DRAW",
        well_filter_client=FakeWellFilterClient(),
    )
    args = {
        "allocation_type": "production",
        "start_date": "2026-06-01",
        "end_date": "2026-06-01",
    }

    summary = json.loads(tool_by_name(tools, "summarize_well_allocation").invoke(args))
    listing = json.loads(tool_by_name(tools, "list_well_allocation").invoke(args))

    assert summary["matched_wells"] == ["HARTZOG DRAW UNIT 1001"]
    assert summary["totals"]["total_oil"] == 10.0
    assert summary["report_flags"] == [
        "Oil production on 1 day(s) at Battery 1 had no producing well with a test rate"
    ]
    assert [row["name"] for row in listing["rows"]] == ["HARTZOG DRAW UNIT 1001"]
    assert listing["rows"][0]["flags"] == ["Well test up to 90 days old used (limit 60)"]
    assert listing["report_flags"] == summary["report_flags"]


def test_list_well_allocation_applies_well_filters():
    tools = build_report_tools(
        FakeAllocationReportClient(),
        4,
        "HARTZOG DRAW",
        well_filter_client=FakeWellFilterClient(),
    )

    result = json.loads(
        tool_by_name(tools, "list_well_allocation").invoke(
            {
                "allocation_type": "production",
                "start_date": "2026-06-01",
                "end_date": "2026-06-01",
                "well_filters": [{"field": "pump_type", "value": "ROD"}],
                "sort_by": "total_oil",
            }
        )
    )

    assert [row["name"] for row in result["rows"]] == [
        "HARTZOG DRAW UNIT 1001",
        "HARTZOG DRAW UNIT 1002",
    ]
