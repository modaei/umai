import json

import httpx
import pytest

from umai.clients.report_client import ReportClient, ReportClientError


def make_client(max_report_days: int = 366) -> ReportClient:
    return ReportClient("http://127.0.0.1:50008/report/", 10, max_report_days)


def test_rejects_unknown_report():
    with pytest.raises(ReportClientError, match="Unsupported report"):
        make_client()._validate_request(1, "unknown", "2026-06-01", "2026-06-02")


def test_rejects_invalid_date_order():
    with pytest.raises(ReportClientError, match="cannot be after"):
        make_client()._validate_request(
            1, "oil_production", "2026-06-03", "2026-06-01"
        )


def test_rejects_excessive_date_range():
    with pytest.raises(ReportClientError, match="cannot exceed 7 days"):
        make_client(max_report_days=7)._validate_request(
            1, "oil_production", "2026-06-01", "2026-06-08"
        )


def test_accepts_valid_request():
    make_client()._validate_request(
        2, "water_injection", "2026-06-01", "2026-06-11"
    )


def test_response_detail_reads_fastapi_error():
    response = httpx.Response(
        503,
        content=json.dumps({"detail": "Report service is busy"}),
        headers={"content-type": "application/json"},
    )
    assert ReportClient._response_detail(response) == "Report service is busy"
