from __future__ import annotations

import logging
import json
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, model_validator

from umai.clients.shutdown_client import ShutdownClient, ShutdownClientError
from umai.tools.rag_enrichment import (
    OperationalContextSearchStore,
    add_operational_context,
)


logger = logging.getLogger(__name__)

class GetShutdownsInput(BaseModel):
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")


class CurrentShutdownsInput(BaseModel):
    as_of_date: str | None = Field(
        default=None,
        description="Optional date in YYYY-MM-DD format. Defaults to today.",
    )


class ActiveWellsInput(BaseModel):
    active_date: str = Field(description="Date in YYYY-MM-DD format.")


class WellFilterInput(BaseModel):
    field: Literal[
        "pump_type",
        "onrr_code",
        "wogcc_class",
        "wogcc_status",
        "direction",
        "prod_fm",
        "battery",
        "lact",
        "monitored",
        "disable_reading",
        "multiple_injection_form",
    ] = Field(description="Allowlisted well attribute to filter by.")
    value: Any = Field(description="Filter value, for example 'Battery 6' or 'rod'.")


class ProducingWellsInput(BaseModel):
    producing_date: str | None = Field(
        default=None, description="Single date in YYYY-MM-DD format."
    )
    start_date: str | None = Field(
        default=None, description="Range start date in YYYY-MM-DD format."
    )
    end_date: str | None = Field(
        default=None, description="Range end date in YYYY-MM-DD format."
    )
    range_mode: Literal["any_day"] = Field(
        default="any_day",
        description="For ranges, include wells producing on any day in the range.",
    )
    filters: list[WellFilterInput] = Field(
        default_factory=list,
        description=(
            "Optional well filters. Use for scoped questions such as Battery 6 "
            "or rod wells. Examples: [{'field':'battery','value':'Battery 6'}], "
            "[{'field':'pump_type','value':'rod'}]."
        ),
    )

    @model_validator(mode="after")
    def validate_date_mode(self) -> "ProducingWellsInput":
        """Require either a single date or a complete date range, never both."""
        has_single = self.producing_date is not None
        has_range = self.start_date is not None or self.end_date is not None
        if has_single == has_range:
            raise ValueError("Provide producing_date or start_date and end_date.")
        if has_range and (self.start_date is None or self.end_date is None):
            raise ValueError("Both start_date and end_date are required.")
        return self


class ShutdownCauseSummaryInput(BaseModel):
    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")


def _json_result(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def build_shutdown_tools(
    client: ShutdownClient,
    site_id: int,
    operational_context_store: OperationalContextSearchStore | None = None,
) -> list[StructuredTool]:
    def list_downtime_codes() -> str:
        """List downtime code descriptions for well shutdowns."""
        return _json_result(client.list_downtime_codes())

    def get_well_shutdowns(
        start_date: str,
        end_date: str,
    ) -> str:
        """Get well shutdowns for the selected site and date range."""
        logger.info(
            "Getting well shutdowns site_id=%s start=%s end=%s",
            site_id,
            start_date,
            end_date,
        )
        try:
            result = {
                "ok": True,
                **client.get_shutdowns(site_id, start_date, end_date),
            }
            enriched = add_operational_context(
                result,
                operational_context_store,
                site_id=site_id,
                query="shutdown downtime context",
                start_date=start_date,
                end_date=end_date,
                limit=8,
            )
            return _json_result(enriched)
        except ShutdownClientError as exc:
            logger.warning("Well shutdown lookup failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def get_current_shutdowns(as_of_date: str | None = None) -> str:
        """Get ongoing shutdowns for the selected site."""
        logger.info(
            "Getting current shutdowns site_id=%s as_of=%s",
            site_id,
            as_of_date,
        )
        try:
            result = {
                "ok": True,
                **client.get_current_shutdowns(site_id, as_of_date),
            }
            as_of = result.get("as_of_date")
            enriched = add_operational_context(
                result,
                operational_context_store,
                site_id=site_id,
                query="current ongoing shutdown context",
                start_date=as_of,
                end_date=as_of,
                limit=8,
            )
            return _json_result(enriched)
        except ShutdownClientError as exc:
            logger.warning("Current shutdown lookup failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def get_active_wells(active_date: str) -> str:
        """Count wells active on one day using only effective ONRR state."""
        logger.info("Getting active wells site_id=%s date=%s", site_id, active_date)
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.get_active_wells(site_id, active_date),
                }
            )
        except ShutdownClientError as exc:
            logger.warning("Active well lookup failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def get_producing_wells(
        producing_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        range_mode: str = "any_day",
        filters: list[dict[str, Any]] | None = None,
    ) -> str:
        """List producing wells for one day or any day in a date range."""
        logger.info(
            "Getting producing wells site_id=%s date=%s start=%s end=%s filters=%s",
            site_id,
            producing_date,
            start_date,
            end_date,
            filters,
        )
        try:
            filter_payload = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in (filters or [])
            ]
            result = (
                client.get_producing_wells(
                    site_id,
                    producing_date,
                    filters=filter_payload,
                )
                if producing_date is not None
                else client.get_producing_wells_for_range(
                    site_id,
                    str(start_date),
                    str(end_date),
                    range_mode,
                    filters=filter_payload,
                )
            )
            return _json_result(
                {
                    "ok": True,
                    **result,
                }
            )
        except ShutdownClientError as exc:
            logger.warning("Producing well lookup failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def summarize_shutdown_causes(
        start_date: str,
        end_date: str,
    ) -> str:
        """Summarize and rank shutdown causes for the selected site and date range."""
        logger.info(
            "Summarizing shutdown causes site_id=%s start=%s end=%s",
            site_id,
            start_date,
            end_date,
        )
        try:
            result = {
                "ok": True,
                **client.summarize_shutdown_causes(site_id, start_date, end_date),
            }
            main_cause = result.get("main_cause") or {}
            query = "shutdown downtime causes context"
            if main_cause.get("downtime_reason"):
                query = f"{query} {main_cause['downtime_reason']}"
            enriched = add_operational_context(
                result,
                operational_context_store,
                site_id=site_id,
                query=query,
                start_date=start_date,
                end_date=end_date,
                limit=10,
            )
            return _json_result(enriched)
        except ShutdownClientError as exc:
            logger.warning("Shutdown cause summary failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    return [
        StructuredTool.from_function(
            func=list_downtime_codes,
            name="list_downtime_codes",
            description="List well shutdown downtime codes and their meanings.",
        ),
        StructuredTool.from_function(
            func=get_well_shutdowns,
            name="get_well_shutdowns",
            description=(
                "Get well shutdowns for a date range. "
                "Use this for downtime, shutdown, shut-in, or reactivation questions."
            ),
            args_schema=GetShutdownsInput,
        ),
        StructuredTool.from_function(
            func=get_current_shutdowns,
            name="get_current_shutdowns",
            description=(
                "Get wells currently on shutdown as of a date. "
                "Use this for current or ongoing shutdown questions."
            ),
            args_schema=CurrentShutdownsInput,
        ),
        StructuredTool.from_function(
            func=get_active_wells,
            name="get_active_wells",
            description=(
                "Count and list active wells for one date. Use this for questions "
                "asking how many wells are active, inactive, online, "
                "or available on a specific day. A well is active only when its "
                "ONRR code is active_well=true as of that day. Shutdown state is "
                "not considered for active-well counts. If the user asks for active "
                "wells with another condition, such as shutdowns, alarms, readings, "
                "tests, or production, use this tool only to get the active-well "
                "population and combine it with the matching domain tool."
            ),
            args_schema=ActiveWellsInput,
        ),
        StructuredTool.from_function(
            func=get_producing_wells,
            name="get_producing_wells",
            description=(
                "Count and list producing wells for one date or date range. For "
                "ranges, any_day includes wells producing on at least one day. "
                "Use this for questions asking how many wells are producing. "
                "When the user scopes the population, pass filters such as "
                "battery=Battery 6 or pump_type=rod; do not return a site-wide "
                "count for a filtered question. "
                "A well is producing only when its ONRR code is active_well=true "
                "and injection_well=false as of that day, and it is not shut down "
                "for the full day. ONRR code name and description are returned "
                "to explain non-producing wells such as injection wells."
            ),
            args_schema=ProducingWellsInput,
        ),
        StructuredTool.from_function(
            func=summarize_shutdown_causes,
            name="summarize_shutdown_causes",
            description=(
                "Summarize and rank shutdown causes for a date range. Use this "
                "when the user asks for the main, top, most common, or biggest "
                "cause/reason for shutdowns or downtime."
            ),
            args_schema=ShutdownCauseSummaryInput,
        ),
    ]
