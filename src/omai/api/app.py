from __future__ import annotations

import logging
import json
from collections.abc import Callable
from datetime import date, datetime, timezone
from threading import BoundedSemaphore
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from omai.rod_pump_health.reader import WellAnalysisRequest, analyzer_from_settings
from omai.rod_pump_health.conclusion import conclusion_analyzer_from_settings
from omai.rod_pump_health.parameter_trends import (
    parameter_analyzer_from_settings,
)

from omai.api.schemas import (
    ChatRequest,
    ChatResponse,
    RodPumpHealthReportRequest,
    WeeklyOverviewRequest,
    WeeklyOverviewResponse,
)
from omai.clients.rod_pump_analysis_client import RodPumpAnalysisClient
from omai.repositories.conversation_repository import (
    ConversationNotFoundError,
    ConversationRepository,
)
from omai.agents.chat_agent import reasoning_effort_for_response_mode
from omai.repositories.daily_usage_repository import (
    DailyUsageLimitExceeded,
    DailyUsageRepository,
)
from omai.config.logging import configure_logging
from omai.config.settings import Settings
from omai.services.chat_service import answer_chat
from omai.services.domain_guard import (
    OUT_OF_DOMAIN_RESPONSE,
    assess_request_integrity,
    is_in_domain,
)
from omai.services.weekly_overview_service import generate_weekly_overview
from omai.services.rag_sources import extract_rag_sources


ALLOWED_HOSTS = {"127.0.0.1", "::1"}

# Temporarily pause the extra reviews while retaining their implementation and saved results.
EXTENDED_HEALTH_ANALYSIS_ENABLED = True

logger = logging.getLogger(__name__)

ChatHandler = Callable[
    [Settings, int, str | None, list[dict[str, str]], str, str],
    tuple[str, list[dict[str, Any]], dict[str, Any]],
]
RodPumpReportHandler = Callable[
    [Settings, int, str | None, list[int] | None],
    dict[str, Any],
]
WeeklyOverviewHandler = Callable[
    [Settings, int, str, str],
    dict[str, str],
]


def create_app(
    settings: Settings | None = None,
    chat_handler: ChatHandler = answer_chat,
    conversation_repository: ConversationRepository | None = None,
    daily_usage_repository: DailyUsageRepository | None = None,
    rod_pump_report_handler: RodPumpReportHandler | None = None,
    weekly_overview_handler: WeeklyOverviewHandler | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)
    app = FastAPI(title="ometrics-ai")
    app.state.settings = settings
    app.state.chat_handler = chat_handler
    app.state.conversation_repository = conversation_repository
    app.state.daily_usage_repository = daily_usage_repository
    app.state.rod_pump_report_handler = (
        rod_pump_report_handler or _run_rod_pump_health_report
    )
    app.state.weekly_overview_handler = (
        weekly_overview_handler or _run_weekly_overview
    )
    app.state.dynograph_slots = BoundedSemaphore(1)
    app.state.chat_slots = BoundedSemaphore(settings.omai_max_concurrent)
    app.state.rod_pump_report_slots = BoundedSemaphore(1)
    app.state.weekly_overview_slots = BoundedSemaphore(1)

    @app.middleware("http")
    async def allow_only_localhost(request: Request, call_next):
        client = request.client
        if not client or not is_allowed_client_host(client.host):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Access allowed only from localhost",
                },
            )
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/rod-pump-health-analysis")
    def rod_pump_health_analysis(payload: WellAnalysisRequest) -> dict[str, Any]:
        """Review cards; run the retained optional stages only when explicitly re-enabled."""
        if not app.state.dynograph_slots.acquire(timeout=settings.omai_slot_timeout):
            raise HTTPException(status_code=503, detail="Rod-pump health analysis is busy")
        try:
            analyzer = analyzer_from_settings(settings)
            well = analyzer._well(payload.well_id)
            try:
                result = analyzer.analyze(payload.well_id, payload.recent_card)
            except Exception:
                logger.exception("Dynograph analysis failed well_id=%s", payload.well_id)
                result = {
                    "well_id": payload.well_id, "site_id": well["site_id"],
                    "review_status": "failed", "review_error": "Dynograph analysis failed.",
                }
            if not EXTENDED_HEALTH_ANALYSIS_ENABLED:
                return {
                    **result, "parameter_trends": {"status": "disabled"},
                    "conclusion": {"status": "disabled"},
                }
            try:
                parameter_trends = parameter_analyzer_from_settings(settings).analyze(payload.well_id)
            except Exception:
                logger.exception("Parameter trend analysis failed well_id=%s", payload.well_id)
                parameter_trends = {
                    "well_id": payload.well_id, "site_id": well["site_id"],
                    "status": "failed", "last_refresh_error": "Parameter trend analysis failed.",
                }
            try:
                conclusion = conclusion_analyzer_from_settings(settings).analyze(
                    payload.well_id, result, parameter_trends,
                )
            except Exception:
                logger.exception("Conclusion analysis failed well_id=%s", payload.well_id)
                conclusion = {
                    "well_id": payload.well_id, "site_id": well["site_id"],
                    "status": "failed", "last_refresh_error": "Conclusion analysis failed.",
                }
            return {**result, "parameter_trends": parameter_trends, "conclusion": conclusion}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Rod-pump health analysis failed")
            raise HTTPException(status_code=503, detail="Rod-pump health analysis failed") from exc
        finally:
            app.state.dynograph_slots.release()

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        if not app.state.chat_slots.acquire(timeout=settings.omai_slot_timeout):
            raise HTTPException(status_code=503, detail="Omai service is busy")

        try:
            repository = (
                app.state.conversation_repository
                or ConversationRepository.from_settings(settings)
            )
            usage_repository = (
                app.state.daily_usage_repository
                or (
                    DailyUsageRepository(
                        repository.engine,
                        request_limit=settings.omai_daily_user_limit_requests,
                        timezone_name=settings.timezone,
                        enabled=settings.omai_daily_user_limit_enabled,
                    )
                    if app.state.conversation_repository is not None
                    else DailyUsageRepository.from_settings(settings)
                )
            )
            conversation_id = payload.conversation_id_text()
            if conversation_id:
                conversation = repository.get(
                    conversation_id,
                    payload.user_id,
                    payload.site_id,
                )
            else:
                conversation = None
            integrity = assess_request_integrity(payload.message)
            reasoning_effort = reasoning_effort_for_response_mode(payload.response_mode)
            if not integrity.allowed:
                conversation = conversation or repository.create(
                    payload.user_id,
                    payload.site_id,
                )
                repository.append_message(conversation, "user", payload.message)
                assistant_message_id = repository.append_message(
                    conversation,
                    "assistant",
                    integrity.response,
                    reasoning_effort=reasoning_effort,
                    info={"request_guard": {"reason_codes": list(integrity.reason_codes)}},
                )
                return ChatResponse(
                    conversation_id=conversation.uuid,
                    answer=integrity.response,
                    assistant_message_id=assistant_message_id,
                )

            usage_repository.consume(payload.user_id, payload.site_id)
            conversation = conversation or repository.create(
                payload.user_id,
                payload.site_id,
            )
            history = repository.load_history(conversation)
            if not history and not is_in_domain(payload.message):
                repository.append_message(conversation, "user", payload.message)
                assistant_message_id = repository.append_message(
                    conversation,
                    "assistant",
                    OUT_OF_DOMAIN_RESPONSE,
                    reasoning_effort=reasoning_effort,
                )
                return ChatResponse(
                    conversation_id=conversation.uuid,
                    answer=OUT_OF_DOMAIN_RESPONSE,
                    assistant_message_id=assistant_message_id,
                )
            handler_args = (
                settings, payload.site_id, None, history, payload.message,
                payload.response_mode,
            )
            if app.state.chat_handler is answer_chat:
                answer, tool_calls, stats = app.state.chat_handler(
                    *handler_args,
                    current_date=(payload.current_date.isoformat() if payload.current_date else None),
                )
            else:
                answer, tool_calls, stats = app.state.chat_handler(*handler_args)
            repository.append_message(conversation, "user", payload.message)
            assistant_message_id = repository.append_message(
                conversation,
                "assistant",
                answer,
                reasoning_effort=reasoning_effort,
                info=_assistant_info(tool_calls, stats, payload.response_mode),
            )
            return ChatResponse(
                conversation_id=conversation.uuid,
                answer=answer,
                assistant_message_id=assistant_message_id,
                data_entry_intent=_data_entry_intent(tool_calls),
                view_intent=_view_intent(payload.message, tool_calls),
            )
        except ConversationNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DailyUsageLimitExceeded as exc:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": str(exc),
                    "reset_at": exc.reset_at.isoformat(),
                },
                headers={"Retry-After": str(_retry_after_seconds(exc.reset_at))},
            ) from exc
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Chat request failed")
            raise HTTPException(status_code=503, detail="Chat request failed") from exc
        finally:
            app.state.chat_slots.release()

    @app.post("/rod-pump-health-report")
    def rod_pump_health_report(payload: RodPumpHealthReportRequest) -> dict[str, Any]:
        """Run deterministic fleet analysis without creating chat records."""
        if not app.state.rod_pump_report_slots.acquire(
            timeout=settings.omai_slot_timeout
        ):
            raise HTTPException(
                status_code=503,
                detail="A rod-pump health report is already running",
            )
        try:
            return app.state.rod_pump_report_handler(
                settings,
                payload.site_id,
                payload.as_of_time.isoformat() if payload.as_of_time else None,
                payload.well_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Rod-pump health report failed")
            raise HTTPException(
                status_code=503,
                detail="Rod-pump health report failed",
            ) from exc
        finally:
            app.state.rod_pump_report_slots.release()

    @app.post("/weekly-overview", response_model=WeeklyOverviewResponse)
    def weekly_overview(payload: WeeklyOverviewRequest) -> WeeklyOverviewResponse:
        """Generate the scheduled weekly email overview without chat state."""
        if not app.state.weekly_overview_slots.acquire(
            timeout=settings.omai_slot_timeout
        ):
            raise HTTPException(
                status_code=503,
                detail="A weekly overview is already running",
            )
        try:
            result = app.state.weekly_overview_handler(
                settings,
                payload.site_id,
                payload.start_date.isoformat(),
                payload.end_date.isoformat(),
            )
            return WeeklyOverviewResponse.model_validate(result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Weekly overview failed")
            raise HTTPException(
                status_code=503,
                detail="Weekly overview failed",
            ) from exc
        finally:
            app.state.weekly_overview_slots.release()

    return app


def _run_rod_pump_health_report(
    settings: Settings,
    site_id: int,
    as_of_time: str | None,
    well_ids: list[int] | None,
) -> dict[str, Any]:
    """Construct the deterministic client outside the chat/conversation path."""
    return RodPumpAnalysisClient.from_settings(settings).rank_wells(
        site_id,
        as_of_time,
        well_ids=well_ids,
    )


def _run_weekly_overview(
    settings: Settings,
    site_id: int,
    start_date: str,
    end_date: str,
) -> dict[str, str]:
    return generate_weekly_overview(
        settings,
        site_id,
        date.fromisoformat(start_date),
        date.fromisoformat(end_date),
    )


def is_allowed_client_host(host: str) -> bool:
    return host in ALLOWED_HOSTS


def _retry_after_seconds(reset_at: datetime) -> int:
    reset_at_utc = reset_at
    if reset_at_utc.tzinfo is None:
        reset_at_utc = reset_at_utc.replace(tzinfo=timezone.utc)
    seconds = int((reset_at_utc - datetime.now(timezone.utc)).total_seconds())
    return max(1, seconds)


def _assistant_info(
    tool_calls: list[dict[str, Any]],
    stats: dict[str, Any],
    response_mode: str,
) -> dict[str, Any]:
    return {
        "response_mode": response_mode,
        "tool_calls": [
            {
                "tool": str(tool_call.get("tool")),
                "arguments": tool_call.get("arguments", {}),
            }
            for tool_call in tool_calls
            if tool_call.get("tool")
        ],
        "time_statistics": stats,
        "rag_documents": extract_rag_sources(tool_calls),
        "sql_queries": _sql_query_audit(tool_calls),
    }


def _data_entry_intent(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Forward only successful navigation intents; warnings remain chat responses."""

    for call in reversed(tool_calls):
        if call.get("tool") != "prepare_data_entry":
            continue
        try:
            result = json.loads(str(call.get("result", "")))
        except (TypeError, ValueError):
            return None
        if result.get("status") == "ready":
            return result
    return None


def _view_intent(message: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Forward ready view intents only for explicit 'show me' requests."""
    if not message.strip().lower().startswith("show me"):
        return None
    for call in reversed(tool_calls):
        if call.get("tool") == "prepare_data_view":
            try:
                result = json.loads(str(call.get("result", "")))
            except (TypeError, ValueError):
                return None
            if result.get("status") == "ready":
                return result
        if call.get("tool") == "run_report":
            arguments = call.get("arguments", {})
            view_type = {
                "gas_flared": "flared_vent",
                "gas_vented": "flared_vent",
                "gas_flared_vented": "flared_vent",
                "gas_fuel": "fuel_gas",
            }.get(arguments.get("report_name"), arguments.get("report_name"))
            if view_type in {
                "oil_sale", "oil_production", "injection_allocation",
                "production_allocation", "water_production", "water_transfer",
                "flared_vent", "fuel_gas", "battery", "water_injection",
            } and arguments.get("start_date") and arguments.get("end_date"):
                return {
                    "status": "ready",
                    "destination": "report",
                    "view_type": view_type,
                    "filters": {
                        "startDate": arguments["start_date"],
                        "endDate": arguments["end_date"],
                    },
                }
    return None


def _sql_query_audit(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract compact SQL audit records for `ai_messages.info`.

    SQL tool results can contain returned rows or validation payloads. The audit
    stores only the query request metadata and deliberately ignores the result.
    """
    audit_records: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        tool_name = str(tool_call.get("tool", ""))
        if tool_name not in {"draft_operational_sql", "execute_operational_sql"}:
            continue

        arguments = tool_call.get("arguments", {})
        audit_records.append(
            {
                "tool": tool_name,
                "question": arguments.get("question"),
                "sql": arguments.get("sql"),
                "notes": arguments.get("notes"),
            }
        )
    return audit_records


app = create_app()
