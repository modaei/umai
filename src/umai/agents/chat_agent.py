from __future__ import annotations

import json
import re
from datetime import date
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from umai.agents.producing_wells_graph import (
    is_producing_well_test_coverage_request,
)
from umai.prompts.chat_agent import (
    build_authoritative_context_message,
    build_chat_system_message,
    build_sql_draft_valid_message,
    build_sql_executed_message,
    build_sql_failed_message,
    build_sql_success_final_message,
)
from umai.tools.rod_pump_health_tools import HEALTH_TOOL_NAMES, is_health_question


MAX_TOOL_ROUNDS = 6
REASONING_EFFORT_BY_RESPONSE_MODE = {
    "fast": "medium",
    "intelligent": "high",
}


def build_model(
    api_key: str,
    model: str,
    base_url: str,
    reasoning_effort: str | None = None,
) -> ChatOpenAI:
    """Create the deterministic chat model used by the UMAI agent.

    The caller owns provider-specific configuration and passes an OpenAI-compatible
    base URL. `reasoning_effort` is optional because not every OpenRouter model
    supports the same reasoning controls.
    """
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=0,
        timeout=60,
        max_retries=2,
        reasoning_effort=reasoning_effort,
    )


def reasoning_effort_for_response_mode(response_mode: str) -> str:
    """Translate the public response-mode value into a model reasoning setting."""
    return REASONING_EFFORT_BY_RESPONSE_MODE.get(response_mode, "medium")


def answer_chat_question(
    model: ChatOpenAI,
    tools: list[BaseTool],
    site_id: int,
    history: list[dict[str, str]],
    question: str,
    site_name: str | None = None,
    authoritative_context: str | None = None,
    today: date | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Answer one user question with tool calling and bounded agent control flow.

    The function builds the complete system prompt, appends recent conversation
    history, lets the model call tools for at most `MAX_TOOL_ROUNDS`, and returns
    the final answer plus local traces and timing statistics. It also enforces a
    few agent-level invariants that are too important to leave only to prompting:
    SQL drafts are de-duplicated, valid SQL drafts are executed once
    automatically, and successful SQL execution forces a final natural-language
    answer instead of allowing more tool calls.
    """
    started_at = perf_counter()
    effective_today = today or date.today()
    if not is_health_question(question):
        tools = [tool for tool in tools if tool.name not in HEALTH_TOOL_NAMES]
    tool_map = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools, parallel_tool_calls=False)

    messages = [
        build_chat_system_message(
            site_name=site_name,
            site_id=site_id,
            today=effective_today.isoformat(),
        )
    ]

    if authoritative_context:
        messages.append(build_authoritative_context_message(authoritative_context))

    for item in history[-10:]:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))

    messages.append(HumanMessage(content=question))
    traces: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "total_seconds": 0.0,
        "model_seconds": 0.0,
        "tool_seconds": 0.0,
        "model_calls": 0,
        "tool_calls": [],
    }
    force_final_response = False
    seen_sql_drafts: set[str] = set()
    failed_sql_attempts = 0
    block_coverage_sql = is_producing_well_test_coverage_request(
        question,
        history,
    )

    for _ in range(MAX_TOOL_ROUNDS):
        model_started_at = perf_counter()
        response = model.invoke(messages) if force_final_response else model_with_tools.invoke(messages)
        stats["model_seconds"] += perf_counter() - model_started_at
        stats["model_calls"] += 1
        messages.append(response)

        if not response.tool_calls:
            stats["total_seconds"] = perf_counter() - started_at
            return _clean_answer(_message_text(response.content)), traces, _rounded_stats(stats)

        for call in response.tool_calls:
            requested_tool_name = call["name"]
            tool_name = requested_tool_name
            arguments = call.get("args", {})
            trace: dict[str, Any] = {"tool": tool_name, "arguments": arguments}

            tool_started_at = perf_counter()
            if requested_tool_name == "execute_operational_sql" and "draft_operational_sql" in tool_map:
                # Treat model-requested execution as a validation draft first.
                # If validation succeeds, _execute_valid_sql_draft runs the SQL
                # immediately below. If validation fails, the model receives
                # repair feedback instead of burning a round on failed execution.
                tool_name = "draft_operational_sql"
                trace["tool"] = tool_name
                trace["requested_tool"] = requested_tool_name
                trace["validation_before_execute"] = True
            tool = tool_map.get(tool_name)
            if tool is None:
                result = f"Unknown tool: {tool_name}"
            elif block_coverage_sql and requested_tool_name in {
                "draft_operational_sql",
                "execute_operational_sql",
            }:
                result = json.dumps(
                    {
                        "ok": False,
                        "executed": False,
                        "error": (
                            "Operational SQL cannot classify active/producing well "
                            "test coverage. Use the dedicated well-population tool "
                            "and an unfiltered search_well_tests result for the "
                            "same date range."
                        ),
                    },
                    separators=(",", ":"),
                )
                force_final_response = True
            elif _is_repeated_sql_draft(tool_name, arguments, seen_sql_drafts):
                # Repeated SQL drafts were a common source of tool-limit failures.
                # Return a structured failure so the model can stop or use prior
                # results instead of consuming another tool round.
                result = json.dumps(
                    {
                        "ok": False,
                        "executed": False,
                        "validation": {
                            "valid": False,
                            "error": "Repeated SQL draft. Execute the previous valid draft or answer from prior tool results.",
                        },
                    },
                    separators=(",", ":"),
                )
            else:
                try:
                    result = tool.invoke(arguments)
                except Exception as exc:  # LangChain schema errors are user-facing here.
                    result = f"Tool validation failed: {exc}"
            tool_seconds = perf_counter() - tool_started_at
            stats["tool_seconds"] += tool_seconds
            stats["tool_calls"].append(
                {
                    "tool": tool_name,
                    "seconds": tool_seconds,
                }
            )

            trace["result"] = str(result)
            traces.append(trace)
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )
            if _is_valid_operational_sql_draft(tool_name, result):
                # A valid draft is already safe to run. Execute it here so the
                # model cannot spend additional rounds drafting equivalent SQL.
                execution_result = _execute_valid_sql_draft(
                    result,
                    tool_map,
                    stats,
                    traces,
                )
                if execution_result is not None:
                    messages.append(build_sql_executed_message(execution_result))
                    if _is_successful_operational_sql_result(
                        "execute_operational_sql",
                        execution_result,
                    ):
                        messages.append(build_sql_success_final_message())
                        force_final_response = True
                else:
                    messages.append(build_sql_draft_valid_message())
            if _is_failed_operational_sql_result(tool_name, result):
                failed_sql_attempts += 1
                if failed_sql_attempts >= 2:
                    messages.append(build_sql_failed_message())
                    force_final_response = True
            if _is_successful_operational_sql_result(tool_name, result):
                messages.append(build_sql_success_final_message())
                force_final_response = True

    if force_final_response:
        model_started_at = perf_counter()
        response = model.invoke(messages)
        stats["model_seconds"] += perf_counter() - model_started_at
        stats["model_calls"] += 1
        stats["total_seconds"] = perf_counter() - started_at
        return _clean_answer(_message_text(response.content)), traces, _rounded_stats(stats)

    stats["total_seconds"] = perf_counter() - started_at
    return (
        "I could not complete the request within the tool-call limit.",
        traces,
        _rounded_stats(stats),
    )


def _message_text(content: Any) -> str:
    """Extract plain text from LangChain/OpenAI message content variants."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part)
    return str(content)


def _is_successful_operational_sql_result(tool_name: str, result: Any) -> bool:
    """Return whether a tool result is a successful executed SQL payload."""
    if tool_name != "execute_operational_sql":
        return False
    payload = _json_payload(result)
    if payload is None:
        return False
    return payload.get("ok") is True and payload.get("executed") is True


def _is_failed_operational_sql_result(tool_name: str, result: Any) -> bool:
    """Return whether a SQL draft or execution tool returned structured failure."""
    if tool_name not in {"draft_operational_sql", "execute_operational_sql"}:
        return False
    payload = _json_payload(result)
    if payload is None:
        return False
    return payload.get("ok") is False


def _is_valid_operational_sql_draft(tool_name: str, result: Any) -> bool:
    """Return whether a draft SQL tool call validated but did not execute SQL."""
    if tool_name != "draft_operational_sql":
        return False
    payload = _json_payload(result)
    if payload is None:
        return False
    return payload.get("ok") is True and payload.get("executed") is False


def _json_payload(result: Any) -> dict[str, Any] | None:
    """Parse structured JSON tool output, preserving dict payloads."""
    if isinstance(result, dict):
        return result
    if not isinstance(result, str):
        return None
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _is_repeated_sql_draft(
    tool_name: str,
    arguments: dict[str, Any],
    seen_sql_drafts: set[str],
) -> bool:
    """Track SQL drafts and reject exact repeats within one agent turn."""
    if tool_name != "draft_operational_sql":
        return False
    normalized = _normalize_sql_for_loop_guard(str(arguments.get("sql", "")))
    if not normalized:
        return False
    if normalized in seen_sql_drafts:
        return True
    seen_sql_drafts.add(normalized)
    return False


def _execute_valid_sql_draft(
    draft_result: str,
    tool_map: dict[str, BaseTool],
    stats: dict[str, Any],
    traces: list[dict[str, Any]],
) -> str | None:
    """Execute a validated SQL draft through the normal SQL execution tool.

    This keeps SQL execution in the existing tool boundary, so validation,
    database permissions, tracing, and timing behave the same as a model-requested
    `execute_operational_sql` call. `None` means the execution tool is unavailable
    or the draft payload is malformed.
    """
    execute_tool = tool_map.get("execute_operational_sql")
    if execute_tool is None:
        return None
    payload = _json_payload(draft_result)
    if payload is None:
        return None
    arguments = {
        "question": payload.get("question") or "",
        "sql": payload.get("sql") or "",
        "notes": payload.get("notes"),
    }
    if not arguments["sql"]:
        return None

    started_at = perf_counter()
    try:
        result = execute_tool.invoke(arguments)
    except Exception as exc:  # Keep parity with normal tool invocation handling.
        result = f"Tool validation failed: {exc}"
    elapsed = perf_counter() - started_at
    stats["tool_seconds"] += elapsed
    stats["tool_calls"].append(
        {
            "tool": "execute_operational_sql",
            "seconds": elapsed,
        }
    )
    traces.append(
        {
            "tool": "execute_operational_sql",
            "arguments": arguments,
            "result": str(result),
            "auto_executed": True,
        }
    )
    return str(result)


def _normalize_sql_for_loop_guard(sql: str) -> str:
    """Normalize SQL enough to detect exact repeated drafts in one turn."""
    return " ".join(sql.strip().rstrip(";").split()).lower()


def _clean_answer(answer: str) -> str:
    """Apply final text cleanup for recurring model phrasing problems."""
    answer = re.sub(
        r"(?im)^\s*the report did not specify units\.?\s*$\n?",
        "",
        answer,
    )
    answer = re.sub(
        r"(?i)(?:\n\s*)?the report did not specify units\.?\s*$",
        "",
        answer,
    )
    return answer.strip()


def _rounded_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Round timing statistics while preserving tool-call structure."""
    return {
        "total_seconds": round(stats["total_seconds"], 3),
        "model_seconds": round(stats["model_seconds"], 3),
        "tool_seconds": round(stats["tool_seconds"], 3),
        "model_calls": stats["model_calls"],
        "tool_calls": [
            {
                "tool": tool_call["tool"],
                "seconds": round(tool_call["seconds"], 3),
            }
            for tool_call in stats["tool_calls"]
        ],
    }
