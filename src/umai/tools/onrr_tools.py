from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from umai.clients.onrr_client import OnrrClient, OnrrClientError


logger = logging.getLogger(__name__)

OnrrStatus = Literal["all", "active", "producing", "injection", "inactive"]


class ExplainOnrrCodeInput(BaseModel):
    code: str = Field(description="ONRR code name, for example POW, SIW, or WIW.")


class WellOnrrStatusInput(BaseModel):
    well_name: str = Field(description="Well name or distinctive part of the well name.")
    as_of_date: str = Field(description="Date in YYYY-MM-DD format.")


class CountWellsByOnrrStatusInput(BaseModel):
    as_of_date: str = Field(description="Date in YYYY-MM-DD format.")
    status: OnrrStatus = Field(
        default="all",
        description="Which ONRR-derived status group to count.",
    )


def _json_result(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def build_onrr_tools(client: OnrrClient, site_id: int) -> list[StructuredTool]:
    def list_onrr_codes() -> str:
        """List ONRR codes and their active/injection meaning."""
        logger.info("Listing ONRR codes")
        try:
            return _json_result({"ok": True, **client.list_onrr_codes()})
        except OnrrClientError as exc:
            logger.warning("ONRR code list failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def explain_onrr_code(code: str) -> str:
        """Explain a single ONRR code."""
        logger.info("Explaining ONRR code=%s", code)
        try:
            return _json_result({"ok": True, **client.explain_onrr_code(code)})
        except OnrrClientError as exc:
            logger.warning("ONRR code explanation failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def get_well_onrr_status_as_of_date(well_name: str, as_of_date: str) -> str:
        """Get a well's effective ONRR status as of one date."""
        logger.info(
            "Getting well ONRR status site_id=%s well=%s date=%s",
            site_id,
            well_name,
            as_of_date,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.get_well_onrr_status_as_of_date(
                        site_id, well_name, as_of_date
                    ),
                }
            )
        except OnrrClientError as exc:
            logger.warning("Well ONRR status lookup failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    def count_wells_by_onrr_status(
        as_of_date: str,
        status: str = "all",
    ) -> str:
        """Count wells by ONRR-derived active/producing/injection status."""
        logger.info(
            "Counting wells by ONRR status site_id=%s date=%s status=%s",
            site_id,
            as_of_date,
            status,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.count_wells_by_onrr_status(site_id, as_of_date, status),
                }
            )
        except OnrrClientError as exc:
            logger.warning("ONRR status count failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    return [
        StructuredTool.from_function(
            func=list_onrr_codes,
            name="list_onrr_codes",
            description="List ONRR codes and whether each represents active or injection wells.",
        ),
        StructuredTool.from_function(
            func=explain_onrr_code,
            name="explain_onrr_code",
            description=(
                "Explain the meaning of one ONRR code, including active_well, "
                "injection_well, and description."
            ),
            args_schema=ExplainOnrrCodeInput,
        ),
        StructuredTool.from_function(
            func=get_well_onrr_status_as_of_date,
            name="get_well_onrr_status_as_of_date",
            description=(
                "Get a well's effective ONRR code and active/injection status "
                "as of a specific date using well history."
            ),
            args_schema=WellOnrrStatusInput,
        ),
        StructuredTool.from_function(
            func=count_wells_by_onrr_status,
            name="count_wells_by_onrr_status",
            description=(
                "Count wells by ONRR-derived status as of a date. Use this for "
                "questions about ONRR-only active, producing, injection, or "
                "inactive well counts when shutdown state and other operational "
                "conditions are not requested. Do not use this as the only tool for "
                "questions that combine status with shutdowns, alarms, readings, "
                "tests, reports, production, tanks, or work orders."
            ),
            args_schema=CountWellsByOnrrStatusInput,
        ),
    ]
