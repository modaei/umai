from __future__ import annotations

import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from umai.clients.view_navigation_client import (
    OPERATIONAL_VIEWS,
    REPORT_VIEWS,
    ViewNavigationClient,
)


class ViewNavigationInput(BaseModel):
    """Arguments extracted from a user request beginning with 'show me'."""

    view_type: str = Field(description=f"Destination data/report type. Supported values: shutdowns, tank_readings, {', '.join(sorted(REPORT_VIEWS | set(OPERATIONAL_VIEWS)))}. Use shutdowns when the user did not specify short or long.")
    start_date: str | None = Field(default=None, description="Optional inclusive start date in YYYY-MM-DD format.")
    end_date: str | None = Field(default=None, description="Optional inclusive end date in YYYY-MM-DD format.")
    search_text: str | None = Field(default=None, description="Plain text for the destination index search box. Do not interpret it as an entity or battery.")
    status: str | None = Field(default=None, description="Optional work-order status: open, closed, or all.")


def build_view_navigation_tools(client: ViewNavigationClient, site_id: int,
                                current_date: str) -> list[StructuredTool]:
    """Build the read-only tool that prepares validated view navigation."""

    def prepare_data_view(view_type: str, start_date: str | None = None,
                          end_date: str | None = None,
                          search_text: str | None = None,
                          status: str | None = None) -> str:
        try:
            result = client.prepare(
                site_id, view_type, current_date, start_date, end_date,
                search_text, status,
            )
        except ValueError as exc:
            result = {"status": "invalid", "message": str(exc)}
        return json.dumps(result, default=str, separators=(",", ":"))

    return [StructuredTool.from_function(
        func=prepare_data_view,
        name="prepare_data_view",
        description=("Prepare navigation for a request whose trimmed text starts with 'show me'. "
                     "Use report view types for reports and operational view types for existing records. "
                     "Pass text after 'for' as search_text exactly as a user would type it in the index search box. "
                     "Reports always return navigation and default to the last seven days. Existing data returns no_results, one detail record, or a filtered index."),
        args_schema=ViewNavigationInput,
    )]
