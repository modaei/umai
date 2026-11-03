from __future__ import annotations

import json
import logging
import re
from calendar import monthrange
from datetime import date, datetime, timezone
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from umai.clients.report_client import AVAILABLE_REPORTS, ReportClient, ReportClientError
from umai.clients.well_filter_client import (
    UnavailableWellFilterClient,
    WellFilterClient,
    WellFilterClientError,
)
from umai.tools.rag_enrichment import (
    OperationalContextSearchStore,
    add_operational_context,
)


logger = logging.getLogger(__name__)

ReportName = Literal[
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
]


class RunReportInput(BaseModel):
    report_name: ReportName = Field(description="The report to generate.")
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")


class CompareReportInput(BaseModel):
    report_name: ReportName = Field(description="The report metric to compare.")
    first_start_date: str = Field(description="First period start, YYYY-MM-DD.")
    first_end_date: str = Field(description="First period end, YYYY-MM-DD.")
    second_start_date: str = Field(description="Second period start, YYYY-MM-DD.")
    second_end_date: str = Field(description="Second period end, YYYY-MM-DD.")


class MonthlyReportSummaryInput(BaseModel):
    report_name: ReportName = Field(description="The report metric to summarize by month.")
    year: int = Field(description="Calendar year to summarize, for example 2026.")
    battery_name: str | None = Field(
        default=None,
        description=(
            "Optional battery name or number to filter, for example 'Battery 6' "
            "or '6'."
        ),
    )
    value_key: str | None = Field(
        default=None,
        description=(
            "Optional metric key inside each daily row. Leave empty for single-metric "
            "reports such as oil_production, gas_flared, and water_injection."
        ),
    )


class WellAllocationSummaryInput(BaseModel):
    allocation_type: Literal["production", "injection"] = Field(
        description="Use production for oil/water/gas allocation, injection for injection allocation.",
    )
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")
    well_name: str | None = Field(
        default=None,
        description="Optional well name, key, or partial well identifier.",
    )
    well_filters: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Optional well attribute filters. Each filter is {'field': field, 'value': value}. "
            "Supported fields include pump_type, onrr_code, wogcc_class, wogcc_status, "
            "direction, prod_fm, battery, lact, monitored, disable_reading, and "
            "multiple_injection_form. Examples: rod wells => pump_type=ROD; "
            "TA wells => onrr_code=TA."
        ),
    )


class WellAllocationListInput(BaseModel):
    allocation_type: Literal["production", "injection"] = Field(
        description="Use production for oil/water/gas allocation, injection for injection allocation.",
    )
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")
    well_name: str | None = Field(
        default=None,
        description="Optional well name, key, or partial well identifier.",
    )
    well_filters: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional well attribute filters, same format as summarize_well_allocation.",
    )
    sort_by: str | None = Field(
        default=None,
        description=(
            "Optional row metric to sort by, such as total_oil, "
            "daily_avg_oil, total_water, total_gas, "
            "total_sold, total_vented, total_flared, total_fuel, "
            "or total_allocated_injection."
        ),
    )
    sort_direction: Literal["asc", "desc"] = "desc"
    limit: int = Field(default=20, ge=1, le=100)


def _json_result(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _battery_number(value: str) -> str | None:
    match = re.search(r"(\d+)$", value.strip())
    return match.group(1) if match else None


def _matches_battery(candidate: str, requested: str) -> bool:
    candidate_norm = _normalize_name(candidate)
    requested_norm = _normalize_name(requested)
    if candidate_norm == requested_norm:
        return True

    requested_number = _battery_number(requested)
    candidate_number = _battery_number(candidate)
    return requested_number is not None and requested_number == candidate_number


def _row_month(row: dict[str, Any]) -> str | None:
    raw_date = row.get("date")
    if raw_date is None:
        return None

    if isinstance(raw_date, str):
        try:
            return date.fromisoformat(raw_date[:10]).isoformat()[:7]
        except ValueError:
            return None

    if isinstance(raw_date, (int, float)):
        try:
            # UMreports indexed report rows use Unix milliseconds for dates.
            return datetime.fromtimestamp(raw_date / 1000, tz=timezone.utc).strftime(
                "%Y-%m"
            )
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(raw_date, date):
        return raw_date.isoformat()[:7]

    return None


def _month_keys(year: int) -> list[str]:
    return [f"{year}-{month:02d}" for month in range(1, 13)]


def _single_metric_value_key(report_name: str, rows: list[dict[str, Any]]) -> str | None:
    if report_name == "battery":
        return None
    if rows and "value" in rows[0]:
        return "value"
    return None


def _allocation_report_name(allocation_type: str) -> str:
    return "injection_allocation" if allocation_type == "injection" else "production_allocation"


def _allocation_metric_keys(allocation_type: str) -> tuple[str, ...]:
    if allocation_type == "injection":
        return ("total_allocated_injection",)
    return (
        "total_oil",
        "total_water",
        "total_gas",
        "total_sold",
        "total_vented",
        "total_flared",
        "total_fuel",
    )


def _matches_well_name(row_name: str, requested: str) -> bool:
    row_norm = _normalize_name(row_name)
    requested_norm = _normalize_name(requested)
    return requested_norm in row_norm


def _summarize_allocation_rows(
    rows: Any,
    allocation_type: str,
    well_names: set[str] | None = None,
    well_name: str | None = None,
    row_filters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {
            "ok": False,
            "error": "Allocation report result was not a list of well rows.",
        }

    selected_rows = []
    report_flags = _allocation_report_flags(rows)
    for row in rows:
        if not isinstance(row, dict) or not _is_well_allocation_row(row):
            continue
        row_name = str(row.get("name") or "")
        if well_names is not None and row_name not in well_names:
            continue
        if well_name and not _matches_well_name(row_name, well_name):
            continue
        if not _matches_allocation_row_filters(row, row_filters or []):
            continue
        selected_rows.append(row)

    totals = {key: 0.0 for key in _allocation_metric_keys(allocation_type)}
    for row in selected_rows:
        for key in totals:
            value = row.get(key)
            if value is None:
                continue
            try:
                totals[key] += float(value)
            except (TypeError, ValueError):
                continue

    result = {
        "ok": True,
        "matched_well_count": len(selected_rows),
        "matched_wells": [row.get("name") for row in selected_rows if row.get("name")],
        "totals": {key: round(value, 2) for key, value in totals.items()},
    }
    if report_flags:
        result["report_flags"] = report_flags
    return result


def _list_allocation_rows(
    rows: Any,
    allocation_type: str,
    start_date: str,
    end_date: str,
    well_names: set[str] | None = None,
    well_name: str | None = None,
    row_filters: list[dict[str, Any]] | None = None,
    sort_by: str | None = None,
    sort_direction: str = "desc",
    limit: int = 20,
) -> dict[str, Any]:
    if not isinstance(rows, list):
        return {
            "ok": False,
            "error": "Allocation report result was not a list of well rows.",
        }

    day_count = _inclusive_day_count(start_date, end_date)
    selected_rows = []
    report_flags = _allocation_report_flags(rows)
    for row in rows:
        if not isinstance(row, dict) or not _is_well_allocation_row(row):
            continue
        row_name = str(row.get("name") or "")
        if well_names is not None and row_name not in well_names:
            continue
        if well_name and not _matches_well_name(row_name, well_name):
            continue
        if not _matches_allocation_row_filters(row, row_filters or []):
            continue
        selected_rows.append(_compact_allocation_row(row, allocation_type, day_count))

    sort_key = sort_by or _default_allocation_sort_key(allocation_type)
    selected_rows.sort(
        key=lambda row: _numeric_sort_value(row.get(sort_key)),
        reverse=sort_direction != "asc",
    )
    result = {
        "ok": True,
        "matched_well_count": len(selected_rows),
        "sort_by": sort_key,
        "sort_direction": sort_direction,
        "limit": limit,
        "rows": selected_rows[:limit],
    }
    if report_flags:
        result["report_flags"] = report_flags
    return result


def _is_well_allocation_row(row: dict[str, Any]) -> bool:
    # Allocation reports may append an "Unallocated" row for site volume that no
    # well could take; it is not a well and must not count toward well totals.
    return row.get("row_type", "well") == "well"


def _allocation_report_flags(rows: list[Any]) -> list[str]:
    flags: list[str] = []
    for row in rows:
        if isinstance(row, dict) and not _is_well_allocation_row(row):
            flags.extend(str(flag) for flag in row.get("flags") or [])
    return flags


def _matches_allocation_row_filters(
    row: dict[str, Any], row_filters: list[dict[str, Any]]
) -> bool:
    for filter_item in row_filters:
        field = str(filter_item.get("field", "")).strip().lower()
        value = str(filter_item.get("value", "")).strip()
        if field == "onrr_code":
            if str(row.get("onrr_code") or "").lower() != value.lower():
                return False
    return True


def _inclusive_day_count(start_date: str, end_date: str) -> int:
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError:
        return 1
    return max((end - start).days + 1, 1)


def _default_allocation_sort_key(allocation_type: str) -> str:
    return (
        "total_allocated_injection"
        if allocation_type == "injection"
        else "total_oil"
    )


def _numeric_sort_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compact_allocation_row(
    row: dict[str, Any], allocation_type: str, day_count: int
) -> dict[str, Any]:
    compact = {
        "name": row.get("name"),
        "onrr_code": row.get("onrr_code"),
    }
    if row.get("onrr_code_description") is not None:
        compact["onrr_code_description"] = row.get("onrr_code_description")
    if row.get("flags"):
        compact["flags"] = row.get("flags")
    for key in _allocation_metric_keys(allocation_type):
        value = row.get(key)
        if value is None:
            continue
        numeric_value = round(_numeric_sort_value(value), 2)
        compact[key] = numeric_value
        avg_key = key.replace("total_", "daily_avg_", 1)
        compact[avg_key] = round(numeric_value / day_count, 2)
    return compact


def _report_row_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        filter_item
        for filter_item in filters
        if str(filter_item.get("field", "")).strip().lower() == "onrr_code"
    ]


def _database_well_filters(filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        filter_item
        for filter_item in filters
        if str(filter_item.get("field", "")).strip().lower() != "onrr_code"
    ]


def _monthly_summary(
    data: Any, report_name: str, year: int, battery_name: str | None, value_key: str | None
) -> dict[str, Any]:
    date_battery_values = (
        data.get("date_battery_values") if isinstance(data, dict) else None
    )
    if not isinstance(date_battery_values, dict):
        return {
            "ok": False,
            "error": "Report result did not include date_battery_values.",
        }

    selected_battery = None
    if battery_name:
        for candidate in date_battery_values:
            if _matches_battery(candidate, battery_name):
                selected_battery = candidate
                break
        if selected_battery is None:
            return {
                "ok": False,
                "error": f"No battery matched {battery_name!r}.",
                "available_batteries": sorted(date_battery_values),
            }

    batteries = [selected_battery] if selected_battery else sorted(date_battery_values)
    monthly_totals = {month: 0.0 for month in _month_keys(year)}

    resolved_value_key = value_key
    matched_rows = 0
    for battery in batteries:
        rows = date_battery_values.get(battery, [])
        if not isinstance(rows, list):
            continue

        if resolved_value_key is None:
            resolved_value_key = _single_metric_value_key(report_name, rows)
        if resolved_value_key is None:
            return {
                "ok": False,
                "error": (
                    "value_key is required for this report because rows contain "
                    "multiple metrics."
                ),
            }

        for row in rows:
            if not isinstance(row, dict):
                continue
            month = _row_month(row)
            if month not in monthly_totals:
                continue
            value = row.get(resolved_value_key)
            if value is None:
                continue
            try:
                monthly_totals[month] += float(value)
                matched_rows += 1
            except (TypeError, ValueError):
                continue

    return {
        "ok": True,
        "report_name": report_name,
        "year": year,
        "battery_name": selected_battery,
        "value_key": resolved_value_key,
        "monthly_totals": {
            month: round(value, 2) for month, value in monthly_totals.items()
        },
        "matched_daily_rows": matched_rows,
    }


def build_report_tools(
    client: ReportClient,
    site_id: int,
    site_name: str | None = None,
    operational_context_store: OperationalContextSearchStore | None = None,
    well_filter_client: WellFilterClient | UnavailableWellFilterClient | None = None,
) -> list[StructuredTool]:
    site_display_name = site_name or "selected site"

    def list_available_reports() -> str:
        """List report names available to the assistant."""
        return _json_result(
            {"site_name": site_display_name, "reports": AVAILABLE_REPORTS}
        )

    def run_report(report_name: str, start_date: str, end_date: str) -> str:
        """Run one oil-field report for the selected site and date range."""
        logger.info(
            "Running report site_id=%s report=%s start=%s end=%s",
            site_id,
            report_name,
            start_date,
            end_date,
        )
        try:
            data = client.run_report(
                site_id, report_name, start_date, end_date
            )
            return _json_result(
                {
                    "ok": True,
                    "site_name": site_display_name,
                    "report_name": report_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "data": data,
                }
            )
        except ReportClientError as exc:
            logger.warning("Report failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def compare_report_periods(
        report_name: str,
        first_start_date: str,
        first_end_date: str,
        second_start_date: str,
        second_end_date: str,
    ) -> str:
        """Retrieve the same report for two periods so they can be compared."""
        logger.info("Comparing report site_id=%s report=%s", site_id, report_name)
        try:
            first = client.run_report(
                site_id, report_name, first_start_date, first_end_date
            )
            second = client.run_report(
                site_id, report_name, second_start_date, second_end_date
            )
            result = {
                "ok": True,
                "site_name": site_display_name,
                "report_name": report_name,
                "first_period": {
                    "start_date": first_start_date,
                    "end_date": first_end_date,
                    "data": first,
                },
                "second_period": {
                    "start_date": second_start_date,
                    "end_date": second_end_date,
                    "data": second,
                },
            }
            enriched = add_operational_context(
                result,
                operational_context_store,
                site_id=site_id,
                query=f"{report_name} report comparison operations context",
                start_date=min(first_start_date, second_start_date),
                end_date=max(first_end_date, second_end_date),
                limit=10,
            )
            return _json_result(enriched)
        except ReportClientError as exc:
            logger.warning("Report comparison failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def summarize_report_by_month(
        report_name: str,
        year: int,
        battery_name: str | None = None,
        value_key: str | None = None,
    ) -> str:
        """Run one report for a year and aggregate its daily battery rows by month."""
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-{monthrange(year, 12)[1]}"
        logger.info(
            "Summarizing report by month site_id=%s report=%s year=%s battery=%s",
            site_id,
            report_name,
            year,
            battery_name,
        )
        try:
            data = client.run_report(site_id, report_name, start_date, end_date)
            summary = _monthly_summary(
                data=data,
                report_name=report_name,
                year=year,
                battery_name=battery_name,
                value_key=value_key,
            )
            summary.update(
                {
                    "site_name": site_display_name,
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
            enriched = add_operational_context(
                summary,
                operational_context_store,
                site_id=site_id,
                query=f"{report_name} monthly report operations context",
                start_date=start_date,
                end_date=end_date,
                limit=12,
            )
            return _json_result(enriched)
        except ReportClientError as exc:
            logger.warning("Monthly report summary failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def summarize_well_allocation(
        allocation_type: str,
        start_date: str,
        end_date: str,
        well_name: str | None = None,
        well_filters: list[dict[str, Any]] | None = None,
    ) -> str:
        """Run allocation report and summarize totals for matching wells."""
        report_name = _allocation_report_name(allocation_type)
        well_filters = well_filters or []
        db_filters = _database_well_filters(well_filters)
        row_filters = _report_row_filters(well_filters)
        logger.info(
            "Summarizing well allocation site_id=%s report=%s start=%s end=%s well=%s filters=%s",
            site_id,
            report_name,
            start_date,
            end_date,
            well_name,
            well_filters,
        )
        try:
            matching_well_names = None
            matched_well_filter = None
            if db_filters or well_name:
                if well_filter_client is None:
                    raise WellFilterClientError("Well filter client is not available.")
                matched_well_filter = well_filter_client.find_wells(
                    site_id=site_id,
                    well_name=well_name,
                    filters=db_filters,
                )
                matching_well_names = {
                    str(row["name"])
                    for row in matched_well_filter.get("wells", [])
                    if row.get("name")
                }

            data = client.run_report(site_id, report_name, start_date, end_date)
            summary = _summarize_allocation_rows(
                data,
                allocation_type=allocation_type,
                well_names=matching_well_names,
                well_name=well_name if matching_well_names is None else None,
                row_filters=row_filters,
            )
            summary.update(
                {
                    "site_name": site_display_name,
                    "report_name": report_name,
                    "allocation_type": allocation_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "filters": {
                        **({"well_name": well_name} if well_name else {}),
                        **({"well_filters": well_filters} if well_filters else {}),
                    },
                    "well_filter": matched_well_filter,
                }
            )
            return _json_result(summary)
        except (ReportClientError, WellFilterClientError) as exc:
            logger.warning("Well allocation summary failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def list_well_allocation(
        allocation_type: str,
        start_date: str,
        end_date: str,
        well_name: str | None = None,
        well_filters: list[dict[str, Any]] | None = None,
        sort_by: str | None = None,
        sort_direction: str = "desc",
        limit: int = 20,
    ) -> str:
        """Run allocation report and return per-well rows."""
        report_name = _allocation_report_name(allocation_type)
        well_filters = well_filters or []
        db_filters = _database_well_filters(well_filters)
        row_filters = _report_row_filters(well_filters)
        logger.info(
            "Listing well allocation site_id=%s report=%s start=%s end=%s well=%s filters=%s sort=%s",
            site_id,
            report_name,
            start_date,
            end_date,
            well_name,
            well_filters,
            sort_by,
        )
        try:
            matching_well_names = None
            matched_well_filter = None
            if db_filters or well_name:
                if well_filter_client is None:
                    raise WellFilterClientError("Well filter client is not available.")
                matched_well_filter = well_filter_client.find_wells(
                    site_id=site_id,
                    well_name=well_name,
                    filters=db_filters,
                )
                matching_well_names = {
                    str(row["name"])
                    for row in matched_well_filter.get("wells", [])
                    if row.get("name")
                }

            data = client.run_report(site_id, report_name, start_date, end_date)
            result = _list_allocation_rows(
                data,
                allocation_type=allocation_type,
                start_date=start_date,
                end_date=end_date,
                well_names=matching_well_names,
                well_name=well_name if matching_well_names is None else None,
                row_filters=row_filters,
                sort_by=sort_by,
                sort_direction=sort_direction,
                limit=limit,
            )
            result.update(
                {
                    "site_name": site_display_name,
                    "report_name": report_name,
                    "allocation_type": allocation_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "filters": {
                        **({"well_name": well_name} if well_name else {}),
                        **({"well_filters": well_filters} if well_filters else {}),
                    },
                    "well_filter": matched_well_filter,
                }
            )
            return _json_result(result)
        except (ReportClientError, WellFilterClientError) as exc:
            logger.warning("Well allocation listing failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    return [
        StructuredTool.from_function(
            func=list_available_reports,
            name="list_available_reports",
            description="List all oil-field reports supported by the reporting service.",
        ),
        StructuredTool.from_function(
            func=run_report,
            name="run_report",
            description=(
                "Run a production, sales, gas, water, battery, injection, or "
                "allocation report for one date range."
            ),
            args_schema=RunReportInput,
        ),
        StructuredTool.from_function(
            func=compare_report_periods,
            name="compare_report_periods",
            description=(
                "Retrieve the same report for two date ranges for comparison. "
                "Use this instead of making two separate report calls."
            ),
            args_schema=CompareReportInput,
        ),
        StructuredTool.from_function(
            func=summarize_report_by_month,
            name="summarize_report_by_month",
            description=(
                "Run one report for a full calendar year and aggregate daily "
                "battery rows into month totals. Use this for month-by-month "
                "battery comparisons instead of calling run_report once per month."
            ),
            args_schema=MonthlyReportSummaryInput,
        ),
        StructuredTool.from_function(
            func=summarize_well_allocation,
            name="summarize_well_allocation",
            description=(
                "Run production_allocation or injection_allocation and summarize "
                "well-level allocation totals. Use this for questions about how "
                "much a specific well or well group produced or injected. Well "
                "groups can be filtered by attributes such as pump_type, onrr_code, "
                "battery, lact, direction, and status fields. Use pump_type=ROD "
                "for rod wells and onrr_code=TA for TA wells. Attribute filters "
                "are case-insensitive. Do not use well tests for production or "
                "injection allocation totals."
            ),
            args_schema=WellAllocationSummaryInput,
        ),
        StructuredTool.from_function(
            func=list_well_allocation,
            name="list_well_allocation",
            description=(
                "Run production_allocation or injection_allocation and return one "
                "row per matched well. Use this for top/bottom/ranked well "
                "allocation questions such as which well produced the most oil, "
                "top 10 producers, or per-well daily averages. Do not use well "
                "tests for production or injection allocation rankings."
            ),
            args_schema=WellAllocationListInput,
        ),
    ]
