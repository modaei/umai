from umai.clients.work_order_client import WorkOrderClientError
from umai.tools.work_order_tools import build_work_order_tools


class FakeWorkOrderClient:
    def __init__(self):
        self.calls = []

    def summarize_costs(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "metric": kwargs["metric"],
            "work_order_count": 2,
            "primary_total_field": "cost_estimate",
            "primary_total": 375,
            "totals_by_field": {"cost_estimate": 375},
            "value_counts_by_field": {"cost_estimate": 2},
        }


class FailingWorkOrderClient:
    def summarize_costs(self, **kwargs):
        raise WorkOrderClientError("database unavailable")


def tool_by_name(tools, name):
    return next(tool for tool in tools if tool.name == name)


def test_summarize_work_order_costs_tool_calls_client_with_site_scope():
    client = FakeWorkOrderClient()
    tool = tool_by_name(build_work_order_tools(client, site_id=4), "summarize_work_order_costs")

    result = tool.invoke(
        {
            "start_date": "2026-05-01",
            "end_date": "2026-05-31",
            "metric": "estimate",
            "search_text": "motor",
        }
    )

    assert '"ok":true' in result
    assert '"primary_total":375' in result
    assert client.calls[0] == {
        "site_id": 4,
        "start_date": "2026-05-01",
        "end_date": "2026-05-31",
        "metric": "estimate",
        "search_text": "motor",
    }


def test_summarize_work_order_costs_tool_returns_json_errors():
    tool = tool_by_name(
        build_work_order_tools(FailingWorkOrderClient(), site_id=4),
        "summarize_work_order_costs",
    )

    result = tool.invoke(
        {
            "start_date": "2026-05-01",
            "end_date": "2026-05-31",
            "metric": "cost",
        }
    )

    assert result == '{"ok":false,"error":"database unavailable"}'
