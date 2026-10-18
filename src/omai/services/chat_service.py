from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from omai.agents.chat_agent import (
    answer_chat_question,
    build_model,
    reasoning_effort_for_response_mode,
)
from omai.agents.navigation_router import try_answer_navigation_request
from omai.agents.onrr_status_graph import try_answer_onrr_status_question
from omai.agents.producing_wells_graph import (
    prepare_well_population_dependency,
    try_answer_producing_well_question,
)
from omai.agents.production_context_graph import prepare_production_context_dependency
from omai.agents.report_comparison_graph import prepare_report_comparison_dependency
from omai.agents.well_test_graph import (
    prepare_well_test_analysis_dependency,
    try_answer_well_test_analysis,
)
from omai.clients.capability_client import CapabilityClient, UnavailableCapabilityClient
from omai.clients.database_schema_client import (
    DatabaseSchemaClient,
    UnavailableDatabaseSchemaClient,
)
from omai.clients.data_point_client import DataPointClient, UnavailableDataPointClient
from omai.clients.data_entry_client import DataEntryClient
from omai.clients.onrr_client import OnrrClient, UnavailableOnrrClient
from omai.clients.reading_client import ReadingClient, UnavailableReadingClient
from omai.clients.report_client import ReportClient
from omai.clients.shutdown_client import ShutdownClient, UnavailableShutdownClient
from omai.clients.site_client import SiteClient
from omai.clients.well_timeline_client import (
    UnavailableWellTimelineClient,
    WellTimelineClient,
)
from omai.clients.well_filter_client import (
    UnavailableWellFilterClient,
    WellFilterClient,
)
from omai.clients.work_order_client import UnavailableWorkOrderClient, WorkOrderClient
from omai.clients.view_navigation_client import ViewNavigationClient
from omai.config.settings import Settings
from omai.rag.vector_store import (
    UnavailableOperationalContextStore,
    VectorOperationalContextStore,
)
from omai.tools.capability_tools import build_capability_tools
from omai.tools.database_schema_tools import build_database_schema_tools
from omai.tools.data_point_tools import build_data_point_tools
from omai.tools.data_entry_tools import build_data_entry_tools
from omai.tools.onrr_tools import build_onrr_tools
from omai.tools.operational_context_tools import build_operational_context_tools
from omai.tools.reading_tools import build_reading_tools
from omai.tools.report_tools import build_report_tools
from omai.tools.shutdown_tools import build_shutdown_tools
from omai.tools.well_timeline_tools import build_well_timeline_tools
from omai.tools.work_order_tools import build_work_order_tools
from omai.tools.view_navigation_tools import build_view_navigation_tools


logger = logging.getLogger(__name__)


def answer_chat(
    settings: Settings,
    site_id: int,
    site_name: str | None,
    history: list[dict[str, str]],
    question: str,
    response_mode: str = "fast",
    current_date: str | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    settings.validate()
    effective_today = date.fromisoformat(current_date) if current_date else date.today()
    effective_current_date = effective_today.isoformat()
    resolved_site_name = site_name or _site_name_from_db(settings, site_id)

    report_client = ReportClient(
        api_url=settings.omreports_api_url,
        timeout_seconds=settings.omreports_timeout_seconds,
        max_report_days=settings.max_report_days,
    )
    try:
        reading_client = ReadingClient.from_settings(settings)
    except ValueError as exc:
        reading_client = UnavailableReadingClient(str(exc))
    try:
        data_point_client = DataPointClient.from_settings(settings)
    except (ValueError, RuntimeError) as exc:
        data_point_client = UnavailableDataPointClient(str(exc))
    try:
        shutdown_client = ShutdownClient.from_settings(settings)
    except ValueError as exc:
        shutdown_client = UnavailableShutdownClient(str(exc))
    try:
        onrr_client = OnrrClient.from_settings(settings)
    except ValueError as exc:
        onrr_client = UnavailableOnrrClient(str(exc))
    try:
        well_filter_client = WellFilterClient.from_settings(settings)
    except ValueError as exc:
        well_filter_client = UnavailableWellFilterClient(str(exc))
    operational_context_store = _operational_context_store_from_settings(settings)
    try:
        well_timeline_client = WellTimelineClient.from_settings(settings)
        well_timeline_client.operational_context_store = operational_context_store
    except ValueError as exc:
        well_timeline_client = UnavailableWellTimelineClient(str(exc))
    try:
        work_order_client = WorkOrderClient.from_settings(settings)
    except ValueError as exc:
        work_order_client = UnavailableWorkOrderClient(str(exc))
    try:
        capability_client = CapabilityClient(_knowledge_dir())
    except Exception as exc:
        capability_client = UnavailableCapabilityClient(str(exc))
    try:
        database_schema_client = DatabaseSchemaClient.from_settings(
            settings,
            _database_schema_path(),
        )
    except Exception as exc:
        database_schema_client = UnavailableDatabaseSchemaClient(str(exc))
    data_entry_client = DataEntryClient.from_settings(settings)
    view_navigation_client = ViewNavigationClient.from_settings(settings)
    tools = [
        *build_data_entry_tools(
            data_entry_client,
            site_id,
            effective_current_date,
        ),
        *build_view_navigation_tools(
            view_navigation_client,
            site_id,
            effective_current_date,
        ),
        *build_capability_tools(capability_client),
        *build_database_schema_tools(
            database_schema_client,
            site_id,
            max_rows=settings.operational_sql_max_rows,
        ),
        # This tool searches the vector DB derived index for notes/comments,
        # work-order context, shutdown explanations, alarms, and history.
        *build_operational_context_tools(operational_context_store, site_id),
        *build_data_point_tools(data_point_client, site_id),
        *build_report_tools(
            report_client,
            site_id,
            resolved_site_name,
            operational_context_store=operational_context_store,
            well_filter_client=well_filter_client,
        ),
        *build_reading_tools(
            reading_client,
            site_id,
            operational_context_store=operational_context_store,
        ),
        *build_onrr_tools(onrr_client, site_id),
        *build_shutdown_tools(
            shutdown_client,
            site_id,
            operational_context_store=operational_context_store,
        ),
        *build_well_timeline_tools(well_timeline_client, site_id),
        *build_work_order_tools(
            work_order_client,
            site_id,
        ),
    ]
    navigation_answer = try_answer_navigation_request(
        tools=tools,
        question=question,
        today=effective_today,
    )
    if navigation_answer is not None:
        return navigation_answer
    onrr_status_answer = try_answer_onrr_status_question(
        tools=tools,
        question=question,
        history=history,
        today=effective_today,
    )
    if onrr_status_answer is not None:
        return onrr_status_answer
    deterministic_answer = try_answer_producing_well_question(
        tools=tools,
        question=question,
        history=history,
        today=effective_today,
    )
    if deterministic_answer is not None:
        return deterministic_answer
    well_test_answer = try_answer_well_test_analysis(
        tools=tools,
        question=question,
        history=history,
        today=effective_today,
    )
    if well_test_answer is not None:
        return well_test_answer
    dependencies = []
    population_dependency = prepare_well_population_dependency(
        tools=tools,
        question=question,
        today=effective_today,
    )
    if population_dependency is not None:
        dependencies.append(population_dependency)
    production_context_dependency = prepare_production_context_dependency(
        tools=tools,
        question=question,
        history=history,
        site_id=site_id,
        today=effective_today,
        reading_client=reading_client,
        well_filter_client=well_filter_client,
    )
    if production_context_dependency is not None:
        dependencies.append(production_context_dependency)
    report_comparison_dependency = prepare_report_comparison_dependency(
        tools=tools,
        question=question,
        today=effective_today,
    )
    if report_comparison_dependency is not None:
        dependencies.append(report_comparison_dependency)
    well_test_dependency = prepare_well_test_analysis_dependency(
        tools=tools,
        question=question,
        history=history,
        today=effective_today,
    )
    if well_test_dependency is not None:
        dependencies.append(well_test_dependency)
    agent_tools = tools
    authoritative_context = None
    if dependencies:
        authoritative_context = "\n".join(item["context"] for item in dependencies)
        used_tool_names = {item["tool_name"] for item in dependencies}
        agent_tools = [tool for tool in tools if tool.name not in used_tool_names]
    model = build_model(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        reasoning_effort=reasoning_effort_for_response_mode(response_mode),
    )
    answer, traces, stats = answer_chat_question(
        model=model,
        tools=agent_tools,
        site_id=site_id,
        site_name=resolved_site_name,
        history=history,
        question=question,
        authoritative_context=authoritative_context,
        today=effective_today,
    )
    if not dependencies:
        return answer, traces, stats
    dependency_traces = [
        trace for dependency in dependencies for trace in dependency["traces"]
    ]
    combined_stats = stats
    for dependency in reversed(dependencies):
        combined_stats = _merge_chat_stats(dependency["stats"], combined_stats)
    return (
        answer,
        [*dependency_traces, *traces],
        combined_stats,
    )


def _merge_chat_stats(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """Combine prefetch and agent timings into the public statistics shape."""
    return {
        "total_seconds": round(first["total_seconds"] + second["total_seconds"], 3),
        "model_seconds": round(first["model_seconds"] + second["model_seconds"], 3),
        "tool_seconds": round(first["tool_seconds"] + second["tool_seconds"], 3),
        "model_calls": first["model_calls"] + second["model_calls"],
        "tool_calls": [*first["tool_calls"], *second["tool_calls"]],
    }


def _site_name_from_db(settings: Settings, site_id: int) -> str | None:
    try:
        return SiteClient.from_settings(settings).get_site_name(site_id)
    except Exception:
        logger.exception("Site lookup failed")
        return None


def _knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "knowledge" / "capabilities"


def _database_schema_path() -> Path:
    return Path(__file__).resolve().parents[3] / "knowledge" / "database_schema.md"


@lru_cache(maxsize=8)
def _operational_context_store_from_settings(
    settings: Settings,
) -> VectorOperationalContextStore | UnavailableOperationalContextStore:
    try:
        # Operational RAG is optional at runtime. Keep the initialized vector DB
        # engine/embedding wrapper cached so each chat request does not rebuild
        # the same retrieval client.
        return VectorOperationalContextStore.from_settings(settings)
    except Exception as exc:
        return UnavailableOperationalContextStore(str(exc))
