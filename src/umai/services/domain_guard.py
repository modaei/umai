from __future__ import annotations

import re
from dataclasses import dataclass


OUT_OF_DOMAIN_RESPONSE = (
    "I can only help with Ometrics related questions."
)

MAX_CHAT_REQUEST_CHARACTERS = 750


@dataclass(frozen=True)
class RequestIntegrityDecision:
    """Outcome of deterministic chat-request safety and scope validation."""

    reason_codes: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return not self.reason_codes

    @property
    def response(self) -> str:
        """Return a concise refusal that identifies every rejected directive."""
        descriptions = {
            "too_long": f"it exceeds the {MAX_CHAT_REQUEST_CHARACTERS}-character limit",
            "role_directive": "it asks me to adopt a role or persona",
            "prescribed_conclusion": "it requires a predetermined conclusion or recommendation",
        }
        reasons = [descriptions[code] for code in self.reason_codes]
        if len(reasons) == 1:
            return (
                f"I can't process this request because {reasons[0]}. "
                f"Please submit a factual Ometrics question, under {MAX_CHAT_REQUEST_CHARACTERS} characters, that lets the data determine the conclusion."
            )

        joined = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"
        return (
            f"I can't process this request because {joined}. "
            f"Please submit a factual Ometrics question, under {MAX_CHAT_REQUEST_CHARACTERS} characters, that lets the data determine the conclusion."
        )


ROLE_DIRECTIVE_PATTERNS = (
    r"\bact\s+as\b",
    r"\bbehave\s+as\b",
    r"\bassume\s+(?:the\s+)?role(?:\s+of)?\b",
    r"\byou\s+are\s+(?:an?|the)\s+[a-z]",
)

PRESCRIBED_CONCLUSION_PATTERNS = (
    r"\bconclude\s+that\b",
    r"\b(?:prove|argue|demonstrate|show)\s+that\b",
    r"\bmake\s+(?:a\s+)?(?:business\s+|profitability\s+)?case\s+(?:for|that)\b",
    r"\b(?:must|should)\s+(?:conclude|recommend)\b",
    r"\b(?:the\s+)?conclusion\s+(?:must|should|is)\b",
)


def assess_request_integrity(question: str) -> RequestIntegrityDecision:
    """Reject oversized requests and instructions that would bias UMAI's answer.

    This guard intentionally does not use an LLM. It is evaluated before tools
    and model calls, so role-play or conclusion-forcing language cannot turn an
    operational data request into an unsupported commissioned report.
    """
    normalized = " ".join(question.strip().split())
    reason_codes: list[str] = []
    if len(normalized) > MAX_CHAT_REQUEST_CHARACTERS:
        reason_codes.append("too_long")
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in ROLE_DIRECTIVE_PATTERNS):
        reason_codes.append("role_directive")
    if any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in PRESCRIBED_CONCLUSION_PATTERNS
    ):
        reason_codes.append("prescribed_conclusion")

    return RequestIntegrityDecision(tuple(reason_codes))


DOMAIN_TERMS = {
    "allocation",
    "alarm",
    "alarms",
    "api",
    "battery",
    "batteries",
    "bsw",
    "capability",
    "capabilities",
    "chart",
    "charts",
    "control",
    "daily",
    "dashboard",
    "data",
    "device",
    "devices",
    "downtime",
    "email",
    "emails",
    "error",
    "field",
    "flare",
    "flared",
    "flaring",
    "flow",
    "flowmeter",
    "gas",
    "hartzog",
    "hdu",
    "injected",
    "injection",
    "knockout",
    "lact",
    "length",
    "load",
    "meter",
    "metrics",
    "missing",
    "note",
    "notes",
    "oil",
    "umai",
    "ometrics",
    "umreports",
    "onrr",
    "plant",
    "pressure",
    "produced",
    "production",
    "pump",
    "reading",
    "readings",
    "report",
    "reports",
    "rod",
    "run",
    "sale",
    "sales",
    "shutdown",
    "shutdowns",
    "site",
    "tank",
    "task",
    "tasks",
    "temperature",
    "test",
    "tests",
    "ticket",
    "tickets",
    "treater",
    "trend",
    "transfer",
    "transferred",
    "uptime",
    "water",
    "well",
    "wells",
    "work",
}

FOLLOW_UP_TERMS = {
    "again",
    "also",
    "compare",
    "continue",
    "daily",
    "details",
    "export",
    "for",
    "from",
    "graph",
    "june",
    "last",
    "may",
    "month",
    "next",
    "now",
    "previous",
    "same",
    "show",
    "that",
    "them",
    "these",
    "this",
    "today",
    "tomorrow",
    "trend",
    "week",
    "yesterday",
}

# Field users often speak in terse operational shorthand. For example,
# "why is 5248 down?" means "why is the well whose name/number ends in 5248
# shut down?" even though the sentence does not contain the word "well".
# These terms are intentionally separate from DOMAIN_TERMS because words like
# "down" or "offline" are too broad by themselves. They only make a question
# in-domain when paired with a plausible numeric field identifier.
WELL_STATE_TERMS = {
    "down",
    "inactive",
    "offline",
    "producing",
    "shut",
    "shutin",
    "stopped",
}

WELL_STATUS_CONTEXT_TERMS = {
    "cause",
    "current",
    "currently",
    "reason",
    "status",
    "why",
}


def is_in_domain(question: str, history: list[dict[str, str]] | None = None) -> bool:
    tokens = _tokenize(question)
    if not tokens:
        return False

    if re.search(r"\bdata(?:\s|_|-)?points?\b", question, re.IGNORECASE):
        return True

    if tokens & DOMAIN_TERMS:
        return True

    if _looks_like_domain_identifier(question):
        return True

    return bool(_has_domain_history(history) and tokens & FOLLOW_UP_TERMS)


def _has_domain_history(history: list[dict[str, str]] | None) -> bool:
    if not history:
        return False

    recent = history[-6:]
    return any(_tokenize(item.get("content", "")) & DOMAIN_TERMS for item in recent)


def _looks_like_domain_identifier(question: str) -> bool:
    tokens = _tokenize(question)
    return bool(
        re.search(r"\b\d{1,2}-\d{1,2}-\d{1,2}\b", question)
        or re.search(r"\bwell\s*\d+\b", question, flags=re.IGNORECASE)
        or _looks_like_data_point_trend_lookup(question)
        or _looks_like_operational_identifier_lookup(question)
        or _looks_like_bare_well_status_question(question, tokens)
        or _looks_like_well_health_request(question)
    )


def _looks_like_well_health_request(question: str) -> bool:
    """Accept health shorthand for a numbered well, e.g. "rerun health for 3051"."""
    return bool(
        re.search(r"\bhealth\b", question, flags=re.IGNORECASE)
        and re.search(r"\b\d{3,6}\b", question)
    )


def _looks_like_data_point_trend_lookup(question: str) -> bool:
    """Accept telemetry trend requests for numbered facilities or wells."""
    if not re.search(r"\b(?:hdu[_\s-]?)?\d{3,6}[a-z]?\b", question, flags=re.IGNORECASE):
        return False

    return bool(
        re.search(
            r"\b(?:analy[sz]e|show|get|describe)\b.+\btrend\b.+\b(?:of|for|in)\s+(?:well\s+)?(?:hdu[_\s-]?)?\d{3,6}[a-z]?\b",
            question,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_operational_identifier_lookup(question: str) -> bool:
    """Accept common field shorthand for asking about a numbered entity.

    A bare number is still not enough. This covers phrases users use for RAG
    lookup, such as "what can you tell me about 4293" or "what happened with
    5144H", while keeping math/age questions out of domain.
    """
    if not re.search(r"\b(?:hdu[_\s-]?)?\d{3,6}[a-z]?\b", question, flags=re.IGNORECASE):
        return False

    return bool(
        re.search(
            r"\b(?:what\s+can\s+you\s+tell\s+me\s+about|tell\s+me\s+about|what\s+happened\s+(?:with|to)|summarize|summary\s+for)\s+(?:hdu[_\s-]?)?\d{3,6}[a-z]?\b",
            question,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_bare_well_status_question(question: str, tokens: set[str]) -> bool:
    """Accept terse well-status shorthand without accepting random numbers.

    The product's users commonly identify wells by the trailing numeric part of
    the well name, e.g. "5248". A bare number is not enough to be domain-safe:
    "how old is 5248?" or "what is 5248 divided by 2?" should still be rejected.
    Requiring an operational state term, plus either a diagnostic/status term or
    a direct "is/was <number> <state>" shape, keeps this rule narrow.
    """
    if not re.search(r"\b\d{3,6}\b", question):
        return False

    if not tokens & WELL_STATE_TERMS:
        return False

    if tokens & WELL_STATUS_CONTEXT_TERMS:
        return True

    return bool(
        re.search(
            r"\b(?:is|was|were|are)\s+\d{3,6}\s+(?:down|offline|inactive|shut|shut[-\s]?in|stopped|producing)\b",
            question,
            flags=re.IGNORECASE,
        )
    )


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 1
    }
