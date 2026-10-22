from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from umai.clients.data_entry_client import DataEntryClient, ENTRY_DEFINITIONS


class DataEntryInput(BaseModel):
    """Model-supplied arguments for a read-only data-entry navigation request."""

    entry_type: str = Field(description=f"Destination form type. Supported values: generic_reading, {', '.join(ENTRY_DEFINITIONS)}. Use generic_reading when the user asks for a reading without explicitly identifying the equipment type.")
    entity_name: str | None = Field(
        default=None,
        description=(
            "Equipment or well display name only, when the form belongs to an "
            "entity. Do not include spoken measurement fields or values here."
        ),
    )
    values: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Form values explicitly supplied by the user. Use snake_case field "
            "names and ISO dates. For tank readings, extract fields such as "
            "top_level_feet, top_level_inches, water_level_feet, "
            "water_level_inches, feet, inches, initial_feet, initial_inches, "
            "final_feet, and final_inches."
        ),
    )


def build_data_entry_tools(client: DataEntryClient, site_id: int, current_date: str) -> list[StructuredTool]:
    """Expose data-entry preparation as a site-scoped agent tool."""

    def prepare_data_entry(entry_type: str, entity_name: str | None = None,
                           values: dict[str, Any] | None = None) -> str:
        return json.dumps(client.prepare(site_id, entry_type, entity_name, values or {}, current_date), default=str, separators=(",", ":"))

    return [StructuredTool.from_function(
        func=prepare_data_entry,
        name="prepare_data_entry",
        description=("Prepare navigation to an existing Ometrics operational data-entry form without creating a record. "
                     "Use it whenever the user asks to create, add, enter, or record a reading, well test, fluid level, injection, shutdown, water draw, run ticket, general note, or work order. "
                     "Keep entity_name limited to the equipment name and put explicitly supplied measurements in values. "
                     "Pass only values the user supplied; missing dates default to today. The result resolves entity names and blocks duplicates or ambiguity."),
        args_schema=DataEntryInput,
    )]
