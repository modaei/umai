from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from umai.clients.work_order_client import WorkOrderClient, WorkOrderClientError


logger = logging.getLogger(__name__)

CostMetric = Literal["cost", "estimate", "all"]


class WorkOrderCostSummaryInput(BaseModel):
    """Arguments for exact work order cost and estimate summaries."""

    start_date: str = Field(description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(description="End date in YYYY-MM-DD format.")
    metric: CostMetric = Field(
        default="all",
        description=(
            "Which numeric work order amount to summarize: cost for final_cost, "
            "estimate for cost_estimate, or all for both."
        ),
    )
    search_text: str | None = Field(
        default=None,
        description=(
            "Optional text to match against work order subject, comments, vendor, "
            "or status."
        ),
    )


def _json_result(value: Any) -> str:
    """Serialize tool responses as compact JSON for the chat model."""
    return json.dumps(value, default=str, separators=(",", ":"))


def build_work_order_tools(
    client: WorkOrderClient,
    site_id: int,
) -> list[StructuredTool]:
    """Build LangChain tools for structured work order cost summaries."""

    def summarize_work_order_costs(
        start_date: str,
        end_date: str,
        metric: str = "all",
        search_text: str | None = None,
    ) -> str:
        """Summarize work order final_cost or cost_estimate by exact SQL totals."""
        logger.info(
            "Summarizing work order costs site_id=%s start=%s end=%s metric=%s",
            site_id,
            start_date,
            end_date,
            metric,
        )
        try:
            return _json_result(
                {
                    "ok": True,
                    **client.summarize_costs(
                        site_id=site_id,
                        start_date=start_date,
                        end_date=end_date,
                        metric=metric,
                        search_text=search_text,
                    ),
                }
            )
        except WorkOrderClientError as exc:
            logger.warning("Work order cost summary failed: %s", exc)
            return _json_result({"ok": False, "error": str(exc)})

    return [
        StructuredTool.from_function(
            func=summarize_work_order_costs,
            name="summarize_work_order_costs",
            description=(
                "Calculate exact work order cost or cost-estimate totals for a "
                "date range from final_cost and cost_estimate database fields. "
                "Use this for work order questions asking how much, total cost, "
                "total estimate, sum, amount, or numeric cost statistics. Use "
                "search_text only when the user names a vendor, status, equipment, "
                "or subject term. The result includes both final_cost and "
                "cost_estimate totals so final-cost gaps can be explained with "
                "the available estimate total."
            ),
            args_schema=WorkOrderCostSummaryInput,
        )
    ]
