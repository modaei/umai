import json
from datetime import date

from umai.services.weekly_overview_service import (
    WeeklyOverviewService,
    _change_summary,
    _previous_period,
    _report_total,
    _system_prompt,
)


class FakeReportClient:
    def __init__(self):
        self.calls = []

    def run_report(self, site_id, report_name, start_date, end_date):
        self.calls.append((site_id, report_name, start_date, end_date))
        if report_name == "oil_production" and start_date == "2026-06-24":
            return {"date_totals": {"a": 100, "b": 120}}
        if report_name == "oil_sale" and start_date == "2026-06-24":
            return {"date_totals": {"a": 80, "b": 100}}
        if report_name == "oil_production" and start_date == "2026-06-16":
            return {"date_totals": {"a": 80, "b": 90}}
        if report_name == "oil_sale" and start_date == "2026-06-16":
            return {"date_totals": {"a": 100, "b": 110}}
        if report_name == "oil_production" and start_date == "2026-06-01":
            return {"date_totals": {"a": 3000}}
        if report_name == "oil_sale" and start_date == "2026-06-01":
            return {"date_totals": {"a": 2800}}
        if report_name == "oil_production" and start_date == "2026-05-01":
            return {"date_totals": {"a": 2500}}
        if report_name == "oil_sale" and start_date == "2026-05-01":
            return {"date_totals": {"a": 2900}}
        return {"date_totals": {"a": 1}}


class FakeShutdownClient:
    def get_shutdowns(self, site_id, start_date, end_date):
        return {
            "count": 2,
            "total_hours": 30,
            "shutdowns": [
                {
                    "well": "Well - HDU 1",
                    "start": start_date,
                    "end": end_date,
                    "downtime_code": "ESP",
                    "comments": "Well pumped off",
                },
                {
                    "well": "Well - HDU 2",
                    "start": start_date,
                    "downtime_code": "PRF",
                    "comments": "Paraffin",
                }
            ],
        }

    def summarize_shutdown_causes(self, site_id, start_date, end_date):
        return {
            "main_cause": {
                "downtime_code": "ESP",
                "downtime_reason": "ESP Downhole Problems",
                "total_hours": 12,
            },
            "causes": [
                {
                    "downtime_code": "ESP",
                    "downtime_reason": "ESP Downhole Problems",
                    "total_hours": 12,
                }
            ],
        }


class FakeContextStore:
    def search(self, **kwargs):
        return {
            "matches": [
                {
                    "source_type": "general_note",
                    "event_date": "2026-06-25",
                    "entity_name": "HDU 1",
                    "text": "General note. Comments: ESP troubleshooting continued.",
                },
                {
                    "source_type": "chart_note",
                    "event_date": "2026-06-26",
                    "entity_name": "HDU 2",
                    "text": "Chart note. Comments: ESP follow-up after pump issue.",
                }
            ]
        }


class FakeModel:
    def __init__(self):
        self.payload = None

    def invoke(self, messages):
        self.payload = json.loads(messages[-1].content.split("\n\n", 1)[1])

        class Response:
            content = "Oil production increased with ESP troubleshooting noted."

        return Response()


def test_generate_weekly_overview_builds_weekly_and_month_context():
    model = FakeModel()
    report_client = FakeReportClient()
    service = WeeklyOverviewService(
        report_client=report_client,
        shutdown_client=FakeShutdownClient(),
        operational_context_store=FakeContextStore(),
        model=model,
    )

    result = service.generate(4, date(2026, 6, 24), date(2026, 7, 1))

    assert result.overview == "Oil production increased with ESP troubleshooting noted."
    assert model.payload["period"] == {
        "start_date": "2026-06-24",
        "end_date": "2026-07-01",
    }
    assert model.payload["month_comparison"] is not None
    assert model.payload["shutdowns"]["total_hours"] == 30
    assert model.payload["shutdowns"]["new_shutdowns"][0]["well"] == "Well - HDU 1"
    assert model.payload["shutdowns"]["common_shutdown_comments"][0]["comment"] == "Well pumped off"
    assert model.payload["operational_context"][0]["source_type"] == "general_note"
    assert set(model.payload["weekly_report_comparisons"]) == {"oil_production", "oil_sale"}
    assert set(model.payload["month_comparison"]) == {"oil_production", "oil_sale"}
    assert {call[1] for call in report_client.calls} == {"oil_production", "oil_sale"}
    assert model.payload["executive_snapshot"]["oil_production"]["current_total"] == 220
    assert model.payload["executive_snapshot"]["oil_sale"]["current_total"] == 180
    assert model.payload["executive_snapshot"]["shutdown_summary"] == {
        "shutdown_count": 2,
        "shutdown_hours": 30,
        "new_shutdown_count": 2,
        "ended_shutdown_count": 1,
        "main_cause": {
            "downtime_code": "ESP",
            "downtime_reason": "ESP Downhole Problems",
            "total_hours": 12,
        },
    }
    assert model.payload["production_sales_gap"] == {
        "oil_production_total": 220.0,
        "oil_sale_total": 180.0,
        "gap": 40.0,
        "direction": "production_above_sales",
    }
    assert model.payload["top_operational_drivers"]["shutdown_causes"][0]["downtime_code"] == "ESP"
    assert model.payload["top_operational_drivers"]["shutdown_comments"][0] == {
        "comment": "Well pumped off",
        "count": 1,
    }
    assert {"type": "new_shutdowns", "count": 2} in model.payload["exceptions"]
    assert any(
        item.get("type") == "production_sales_gap"
        for item in model.payload["exceptions"]
    )
    assert any(
        item.get("type") == "repeated_operational_context" and item.get("theme") == "esp"
        for item in model.payload["exceptions"]
    )
    assert ("2026-06-01", "2026-06-30") in {
        (call[2], call[3]) for call in report_client.calls
    }
    instructions = model.payload["instructions"]["format"]
    assert "Operational Focus Areas" in instructions
    assert "Do not include a Key metrics section" in instructions
    assert "During [current period]" in instructions
    assert "compared to the previous week" in instructions
    assert "while oil sale" in instructions
    assert "opposite directions are unambiguous" in instructions
    assert "compared with the previous month" in instructions


def test_weekly_overview_prompt_uses_operational_focus_areas_not_key_metrics():
    prompt = _system_prompt()

    assert "Operational Focus Areas" in prompt
    assert "Do not include a 'Key metrics:' section" in prompt
    assert "During [current period]" in prompt
    assert "compared to the previous week" in prompt
    assert "while oil sale" in prompt
    assert "Do not use 'respectively'" in prompt
    assert "compared with the previous month" in prompt
    assert "previous month date range" in prompt
    assert "'Management attention:'" not in prompt


def test_generate_weekly_overview_omits_month_context_without_boundary():
    model = FakeModel()
    service = WeeklyOverviewService(
        report_client=FakeReportClient(),
        shutdown_client=FakeShutdownClient(),
        operational_context_store=FakeContextStore(),
        model=model,
    )

    service.generate(4, date(2026, 6, 17), date(2026, 6, 23))

    assert model.payload["month_comparison"] is None


def test_report_total_handles_known_report_shapes():
    assert _report_total({"date_totals": {"a": 1, "b": "2.5"}}) == 3.5
    assert _report_total([{"value": 3}, {"value": "4"}]) == 7
    assert _report_total([{"flared": 2}, {"flared": 5}]) == 7


def test_change_summary_describes_direction_and_percent():
    assert _change_summary(120, 100) == {
        "direction": "increase",
        "delta": 20,
        "percent": 20,
    }
    assert _change_summary(80, 100)["direction"] == "decrease"
    assert _change_summary(0, 0)["direction"] == "flat"


def test_previous_period_uses_same_inclusive_day_count():
    assert _previous_period(date(2026, 6, 24), date(2026, 7, 1)) == (
        date(2026, 6, 16),
        date(2026, 6, 23),
    )
