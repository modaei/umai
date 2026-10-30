from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from umai.clients.rod_pump_health_client import (
    RodPumpHealthClient,
    RodPumpHealthError,
)


HEALTH_TOOL_NAMES = frozenset({
    "get_rod_pump_health",
    "list_rod_pump_health",
    "request_rod_pump_health_rerun",
})
RERUN_TOOL_NAME = "request_rod_pump_health_rerun"


def is_health_question(question: str) -> bool:
    """Health tools require the explicit word health in the current request."""
    return bool(re.search(r"\bhealth\b", question, flags=re.IGNORECASE))


class RodPumpWellInput(BaseModel):
    well_name: str = Field(
        description=(
            "Exact well name, exact telemetry key, or complete unique field "
            "identifier in the selected site, such as 5823 or HDU_5823."
        )
    )


class NoInput(BaseModel):
    pass


def _json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


def build_rod_pump_health_tools(
    client: RodPumpHealthClient,
    site_id: int,
) -> list[StructuredTool]:
    """Expose saved, site-bound health results without allowing the model to choose site scope."""

    def get_rod_pump_health(well_name: str) -> str:
        try:
            return _json({"ok": True, **client.well_health(site_id, well_name)})
        except RodPumpHealthError as exc:
            return _json({"ok": False, "error": str(exc)})

    def list_rod_pump_health() -> str:
        try:
            return _json({"ok": True, **client.site_health(site_id)})
        except RodPumpHealthError as exc:
            return _json({"ok": False, "error": str(exc)})

    def request_rod_pump_health_rerun(well_name: str) -> str:
        # umai only records the request. UMetriqs runs the analysis as the
        # signed-in user after this answer is returned.
        try:
            saved = client.well_health(site_id, well_name)
        except RodPumpHealthError as exc:
            return _json({"ok": False, "error": str(exc)})
        well = saved["well"]
        if saved.get("refresh", {}).get("in_progress"):
            return _json({
                "ok": False,
                "well": well,
                "error": f"A health check is already running for {well['name']}.",
            })
        return _json({
            "ok": True,
            "status": "rerun_requested",
            "well": well,
            "message": (
                f"UMetriqs will start a new health check for {well['name']} after this "
                "reply. It can take several minutes and the result is posted to this "
                "conversation when it finishes. Do not describe new results."
            ),
        })

    return [
        StructuredTool.from_function(
            func=get_rod_pump_health,
            name="get_rod_pump_health",
            description=(
                "Use only when the current user question explicitly contains the word health. "
                "Return the saved rod-pump health review for one well, the same result shown on "
                "the UMetriqs rod-pump health page: health status (healthy, watch, problem), "
                "summary, findings with reasoning and recommended actions, surface-load check, "
                "parameter-trend interpretation, conclusion, shutdown or exclusion state, the "
                "time of the reviewed card, and when the review was saved. Prefer this over SQL "
                "for rod-pump health questions."
            ),
            args_schema=RodPumpWellInput,
        ),
        StructuredTool.from_function(
            func=list_rod_pump_health,
            name="list_rod_pump_health",
            description=(
                "Use only when the current user question explicitly contains the word health. "
                "Return the saved health status of every monitored rod-pump well in the site, "
                "ordered problem, watch, healthy, then unavailable or not analyzed, with counts. "
                "Use for site-wide health questions such as which wells need attention."
            ),
            args_schema=NoInput,
        ),
        StructuredTool.from_function(
            func=request_rod_pump_health_rerun,
            name=RERUN_TOOL_NAME,
            description=(
                "Request a new health check for one rod-pump well. Each run costs money. Call it "
                "only when the current user message itself explicitly asks to run, rerun, or "
                "refresh the health check for that specific well. Never call it because saved "
                "results are old, missing, or failed, and never call it for more than one well."
            ),
            args_schema=RodPumpWellInput,
        ),
    ]
