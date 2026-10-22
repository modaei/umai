from __future__ import annotations

import re
from typing import Any

from umai.evals.models import DeterministicCheckResult, EvaluationCase
from umai.services.domain_guard import OUT_OF_DOMAIN_RESPONSE


DATE_ISO_PATTERN = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
SITE_ID_PATTERN = re.compile(r"\bsite[_ ]?id\b|\bsite\s+\d+\b", re.IGNORECASE)
DATABASE_ID_PATTERN = re.compile(r"\b(?:asset_id|entity_id|well_id|tank_id)\b", re.IGNORECASE)


def run_deterministic_checks(
    case: EvaluationCase,
    answer: str,
    tool_calls: list[dict[str, Any]],
) -> list[DeterministicCheckResult]:
    """Run hard checks that should not depend on another LLM judge."""

    checks = []
    tool_names = [str(call.get("tool")) for call in tool_calls if call.get("tool")]
    checks.extend(_forbidden_tool_checks(case, tool_names))
    checks.extend(_expected_tool_call_checks(case, tool_calls))
    checks.extend(_phrase_checks(case, answer))
    checks.extend(_global_answer_checks(case, answer))
    return checks


def _forbidden_tool_checks(
    case: EvaluationCase, tool_names: list[str]
) -> list[DeterministicCheckResult]:
    return [
        DeterministicCheckResult(
            name=f"forbidden_tool:{tool_name}",
            passed=tool_name not in tool_names,
            message=(
                f"Forbidden tool {tool_name!r} was not used."
                if tool_name not in tool_names
                else f"Forbidden tool {tool_name!r} was used."
            ),
        )
        for tool_name in case.forbidden_tools
    ]


def _expected_tool_call_checks(
    case: EvaluationCase, tool_calls: list[dict[str, Any]]
) -> list[DeterministicCheckResult]:
    checks = []
    for index, expectation in enumerate(case.expected_tool_calls):
        if "one_of" in expectation:
            checks.append(_one_of_expected_tool_call_check(index, expectation, tool_calls))
            continue
        checks.append(_single_expected_tool_call_check(index, expectation, tool_calls))
    return checks


def _one_of_expected_tool_call_check(
    index: int, expectation: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> DeterministicCheckResult:
    alternatives = expectation.get("one_of", [])
    failure_messages = []
    labels = []
    for alternative in alternatives:
        passed, message, label = _expected_tool_call_matches(alternative, tool_calls)
        labels.append(label)
        if passed:
            return DeterministicCheckResult(
                name=f"expected_tool_call:{index + 1}:one_of",
                passed=True,
                message=f"Expected tool call #{index + 1} matched one of {labels}.",
            )
        failure_messages.append(message)
    return DeterministicCheckResult(
        name=f"expected_tool_call:{index + 1}:one_of",
        passed=False,
        message=(
            f"Expected tool call #{index + 1} to match one of {labels}. "
            + "; ".join(failure_messages)
        ),
    )


def _single_expected_tool_call_check(
    index: int, expectation: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> DeterministicCheckResult:
    passed, message, label = _expected_tool_call_matches(expectation, tool_calls)
    return DeterministicCheckResult(
        name=f"expected_tool_call:{index + 1}:{label}",
        passed=passed,
        message=message,
    )


def _expected_tool_call_matches(
    expectation: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> tuple[bool, str, str]:
    tool_name = str(expectation["tool"])
    expected_arguments = expectation.get("arguments", {})
    matching_tool_calls = [
        call for call in tool_calls if str(call.get("tool")) == tool_name
    ]
    passed = False
    failure_messages = []
    for call in matching_tool_calls:
        call_passed, message = _tool_arguments_match(
            call.get("arguments") or {}, expected_arguments
        )
        if call_passed:
            passed = True
            break
        failure_messages.append(message)
    if passed:
        message = f"Expected tool call {tool_name!r} matched."
    elif not matching_tool_calls:
        used = [str(call.get("tool")) for call in tool_calls if call.get("tool")]
        message = f"Expected tool call {tool_name!r} was not used. Used: {used}"
    else:
        message = (
            f"Expected tool call {tool_name!r} argument mismatch: "
            + "; ".join(failure_messages)
        )
    return passed, message, tool_name


def _tool_arguments_match(
    actual_arguments: Any, expected_arguments: dict[str, Any]
) -> tuple[bool, str]:
    if not isinstance(actual_arguments, dict):
        return False, f"actual arguments were not an object: {actual_arguments!r}"
    failures = []
    for path, expected_value in expected_arguments.items():
        actual_found, actual_value = _get_path(actual_arguments, str(path))
        matched, reason = _argument_value_matches(actual_found, actual_value, expected_value)
        if not matched:
            failures.append(
                f"{path}: expected {_describe_expected(expected_value)}, "
                f"actual {_describe_actual(actual_found, actual_value)} ({reason})"
            )
    if failures:
        return False, ", ".join(failures)
    return True, "arguments matched"


def _get_path(value: Any, path: str) -> tuple[bool, Any]:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return False, None
            if 0 <= index < len(current):
                current = current[index]
                continue
        return False, None
    return True, current


def _argument_value_matches(
    actual_found: bool, actual_value: Any, expected_value: Any
) -> tuple[bool, str]:
    if isinstance(expected_value, dict) and _is_matcher(expected_value):
        return _matcher_value_matches(actual_found, actual_value, expected_value)
    if not actual_found:
        return False, "path is missing"
    if _numeric_values_equal(actual_value, expected_value):
        return True, "numeric values matched"
    if isinstance(actual_value, str) and isinstance(expected_value, str):
        if actual_value.casefold() == expected_value.casefold():
            return True, "strings matched case-insensitively"
        return False, "strings differ"
    if actual_value == expected_value:
        return True, "values matched"
    return False, "values differ"


def _matcher_value_matches(
    actual_found: bool, actual_value: Any, matcher: dict[str, Any]
) -> tuple[bool, str]:
    if "absent" in matcher:
        expected_absent = bool(matcher["absent"])
        is_absent = not actual_found or actual_value is None
        return (
            is_absent == expected_absent,
            "absence matched" if is_absent == expected_absent else "absence differed",
        )
    if "present" in matcher:
        expected_present = bool(matcher["present"])
        is_present = actual_found and actual_value is not None
        return (
            is_present == expected_present,
            "presence matched" if is_present == expected_present else "presence differed",
        )
    if not actual_found:
        return False, "path is missing"
    if "contains" in matcher:
        actual_text = str(actual_value).casefold()
        expected_text = str(matcher["contains"]).casefold()
        return (
            expected_text in actual_text,
            "substring matched" if expected_text in actual_text else "substring missing",
        )
    if "contains_all" in matcher:
        actual_text = str(actual_value).casefold()
        missing = [
            str(item) for item in matcher["contains_all"]
            if str(item).casefold() not in actual_text
        ]
        return (
            not missing,
            "all substrings matched" if not missing else f"missing substrings: {missing}",
        )
    if "one_of" in matcher:
        options = matcher["one_of"]
        if not isinstance(options, list | tuple):
            return False, "one_of matcher value is not a list"
        for option in options:
            matched, _reason = _argument_value_matches(True, actual_value, option)
            if matched:
                return True, "one_of option matched"
        return False, "no one_of option matched"
    return False, "unknown matcher"


def _is_matcher(value: dict[str, Any]) -> bool:
    return bool(
        {"contains", "contains_all", "one_of", "absent", "present"} & set(value)
    )


def _numeric_values_equal(actual_value: Any, expected_value: Any) -> bool:
    if isinstance(actual_value, bool) or isinstance(expected_value, bool):
        return False
    if not isinstance(actual_value, int | float) or not isinstance(
        expected_value, int | float
    ):
        return False
    return float(actual_value) == float(expected_value)


def _describe_expected(expected_value: Any) -> str:
    return repr(expected_value)


def _describe_actual(actual_found: bool, actual_value: Any) -> str:
    if not actual_found:
        return "<missing>"
    return repr(actual_value)


def _phrase_checks(case: EvaluationCase, answer: str) -> list[DeterministicCheckResult]:
    normalized = answer.casefold()
    checks = []
    for phrase in case.required_phrases:
        found = phrase.casefold() in normalized
        checks.append(
            DeterministicCheckResult(
                name=f"required_phrase:{phrase}",
                passed=found,
                message=(
                    f"Required phrase {phrase!r} was present."
                    if found
                    else f"Required phrase {phrase!r} was missing."
                ),
            )
        )
    for phrase in case.forbidden_phrases:
        found = phrase.casefold() in normalized
        checks.append(
            DeterministicCheckResult(
                name=f"forbidden_phrase:{phrase}",
                passed=not found,
                message=(
                    f"Forbidden phrase {phrase!r} was absent."
                    if not found
                    else f"Forbidden phrase {phrase!r} was present."
                ),
            )
        )
    return checks


def _global_answer_checks(
    case: EvaluationCase, answer: str
) -> list[DeterministicCheckResult]:
    checks = [
        DeterministicCheckResult(
            name="no_site_id_leak",
            passed=SITE_ID_PATTERN.search(answer) is None,
            message="Answer does not expose site_id." if SITE_ID_PATTERN.search(answer) is None else "Answer exposes site_id/site number.",
        ),
        DeterministicCheckResult(
            name="no_database_id_labels",
            passed=DATABASE_ID_PATTERN.search(answer) is None,
            message="Answer does not expose database-only ID labels." if DATABASE_ID_PATTERN.search(answer) is None else "Answer exposes database-only ID labels.",
        ),
    ]
    if "guardrail" in case.category:
        checks.append(
            DeterministicCheckResult(
                name="out_of_domain_refusal",
                passed=answer.strip() == OUT_OF_DOMAIN_RESPONSE,
                message=(
                    "Out-of-domain refusal matched."
                    if answer.strip() == OUT_OF_DOMAIN_RESPONSE
                    else "Out-of-domain refusal did not match."
                ),
            )
        )
    if "data_entry" in case.category or "workflow" in case.category:
        lower = answer.casefold()
        record_created = any(
            phrase in lower
            for phrase in (
                "record has been created",
                "i created",
                "created the reading",
                "saved the reading",
                "submitted the reading",
            )
        )
        checks.append(
            DeterministicCheckResult(
                name="no_create_claim",
                passed=not record_created,
                message=(
                    "Answer does not claim a record was created."
                    if not record_created
                    else "Answer claims a record was created."
                ),
            )
        )
    if DATE_ISO_PATTERN.search(answer):
        checks.append(
            DeterministicCheckResult(
                name="no_iso_dates",
                passed=False,
                message="Answer contains ISO date formatting instead of MM/DD/YYYY.",
            )
        )
    return checks
