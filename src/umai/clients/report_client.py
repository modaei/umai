from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx


AVAILABLE_REPORTS = (
    "battery",
    "gas_flared",
    "gas_flared_vented",
    "gas_fuel",
    "gas_production",
    "gas_vented",
    "injection_allocation",
    "oil_production",
    "oil_sale",
    "production_allocation",
    "water_injection",
    "water_production",
    "water_transfer",
)


class ReportClientError(RuntimeError):
    """Raised when a report request is invalid or cannot be completed."""


class ReportClient:
    def __init__(self, api_url: str, timeout_seconds: float, max_report_days: int):
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.max_report_days = max_report_days

    def run_report(
        self,
        site_id: int,
        report_name: str,
        start_date: str,
        end_date: str,
    ) -> Any:
        self._validate_request(site_id, report_name, start_date, end_date)

        payload = {
            "site_id": site_id,
            "function_name": report_name,
            "params": {"start_date": start_date, "end_date": end_date},
        }

        try:
            response = httpx.post(
                self.api_url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ReportClientError("The reporting service timed out.") from exc
        except httpx.HTTPStatusError as exc:
            detail = self._response_detail(exc.response)
            raise ReportClientError(
                f"The reporting service returned HTTP {exc.response.status_code}: {detail} \n"
                f"url: {self.api_url}, payload: {payload}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ReportClientError(
                "Could not connect to the reporting service. "
                "Confirm that omreports is running on the configured URL."
            ) from exc

        try:
            result = response.json()
            if isinstance(result, str):
                result = json.loads(result)
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            raise ReportClientError(
                "The reporting service returned an invalid JSON response."
            ) from exc

    def _validate_request(
        self,
        site_id: int,
        report_name: str,
        start_date: str,
        end_date: str,
    ) -> None:
        if site_id <= 0:
            raise ReportClientError("site_id must be a positive integer.")
        if report_name not in AVAILABLE_REPORTS:
            raise ReportClientError(f"Unsupported report: {report_name}")

        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
        except ValueError as exc:
            raise ReportClientError("Dates must use YYYY-MM-DD format.") from exc

        if start > end:
            raise ReportClientError("start_date cannot be after end_date.")

        inclusive_days = (end - start).days + 1
        if inclusive_days > self.max_report_days:
            raise ReportClientError(
                f"Report range cannot exceed {self.max_report_days} days."
            )

    @staticmethod
    def _response_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
            if isinstance(body, dict):
                return str(body.get("detail", body))
            return str(body)
        except ValueError:
            return response.text[:500] or "No response body"

