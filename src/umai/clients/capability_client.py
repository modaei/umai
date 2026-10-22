from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CapabilityClientError(RuntimeError):
    """Raised when capability knowledge cannot be loaded or searched."""


@dataclass(frozen=True)
class CapabilityDocument:
    title: str
    source: str
    content: str
    tokens: set[str]


class CapabilityClient:
    def __init__(self, knowledge_dir: Path | str):
        self.knowledge_dir = Path(knowledge_dir)
        # Load once at startup/request construction time. Capability docs are small
        # markdown files, so keeping tokenized copies in memory avoids repeated disk IO.
        self.documents = self._load_documents()

    @classmethod
    def from_default(cls) -> "CapabilityClient":
        return cls(Path.cwd() / "knowledge" / "capabilities")

    def search(self, query: str, limit: int = 3) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise CapabilityClientError("Query cannot be empty.")
        if limit <= 0:
            raise CapabilityClientError("limit must be positive.")

        query_tokens = _tokenize(query)
        # Capability questions often use user-facing words that differ from the
        # internal feature names. Expand those terms before matching against docs.
        expanded_tokens = query_tokens | _expand_query_tokens(query_tokens)
        scored = []
        for document in self.documents:
            overlap = expanded_tokens & document.tokens
            # Exact domain phrases are stronger signals than single-token overlap.
            # This keeps targeted guidance like "production allocation" above more
            # generic report/readings documents.
            phrase_bonus = _phrase_bonus(query, document.content)
            score = len(overlap) + phrase_bonus
            if score <= 0:
                continue
            scored.append((score, document, overlap))

        scored.sort(key=lambda item: (-item[0], item[1].title))
        matches = []
        for score, document, overlap in scored[:limit]:
            matches.append(
                {
                    "capability": document.title,
                    "source": document.source,
                    "score": score,
                    "matched_terms": sorted(overlap)[:12],
                    "summary": _summary(document.content),
                }
            )

        return {
            "query": query,
            "count": len(matches),
            "matches": matches,
        }

    def _load_documents(self) -> list[CapabilityDocument]:
        if not self.knowledge_dir.exists():
            raise CapabilityClientError(
                f"Capability knowledge directory does not exist: {self.knowledge_dir}"
            )

        documents = []
        for path in sorted(self.knowledge_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            # Tokenize the full document once; search only needs set overlap and a
            # short summary, not a heavier vector index for this small knowledge base.
            documents.append(
                CapabilityDocument(
                    title=_title_from_content(content, path),
                    source=str(path),
                    content=content,
                    tokens=_tokenize(content),
                )
            )

        if not documents:
            raise CapabilityClientError(
                f"No capability documents found in {self.knowledge_dir}"
            )
        return documents


class UnavailableCapabilityClient:
    def __init__(self, reason: str):
        self.reason = reason

    def search(self, query: str, limit: int = 3) -> dict[str, Any]:
        # Allows agent/tool wiring to fail fast with a clear reason when capability
        # documents are missing, while preserving the same public search interface.
        raise CapabilityClientError(self.reason)


def _title_from_content(content: str, path: Path) -> str:
    first_line = content.splitlines()[0].strip()
    if first_line.startswith("# "):
        return first_line.removeprefix("# ").strip()
    return path.stem.replace("_", " ").title()


def _tokenize(text: str) -> set[str]:
    # Keep tokenization deliberately simple and deterministic. The capability
    # corpus is curated markdown, so lowercased alphanumeric tokens are enough.
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 2 and token not in STOP_WORDS
    }


def _expand_query_tokens(tokens: set[str]) -> set[str]:
    expanded = set()
    for token in tokens:
        expanded.update(SYNONYMS.get(token, set()))
    return expanded


def _phrase_bonus(query: str, content: str) -> int:
    query = query.lower()
    content = content.lower()
    bonus = 0
    for phrase in KEY_PHRASES:
        if phrase in query and phrase in content:
            bonus += 4
    return bonus


def _summary(content: str, max_chars: int = 5_000) -> str:
    # Return compact context for the LLM/tool result. Headings are metadata and the
    # cap prevents capability guidance from crowding out the user's actual request.
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.startswith("# ")
    ]
    text = " ".join(lines)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


STOP_WORDS = {
    "about",
    "after",
    "and",
    "are",
    "can",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "that",
    "the",
    "this",
    "use",
    "user",
    "when",
    "where",
    "which",
    "with",
}

SYNONYMS = {
    "down": {"shutdown", "shut", "offline", "downtime"},
    "offline": {"shutdown", "down", "downtime"},
    "register": {"record", "enter", "track"},
    "injected": {"injection", "inject"},
    "injecting": {"injection", "inject"},
    "produced": {"production", "produce"},
    "sold": {"sale", "sales"},
    "flared": {"flare", "flaring"},
    "transferred": {"transfer"},
    "contributed": {"allocation", "attribute", "production"},
    "contribution": {"allocation", "attribute", "production"},
    "well": {"wells"},
    "email": {"emails", "summary", "summaries"},
    "alert": {"alarm", "alarms"},
    "alerts": {"alarm", "alarms"},
    "job": {"task", "work", "order"},
    "maintenance": {"task", "work", "order"},
}

KEY_PHRASES = {
    "well is down",
    "well down",
    "shut in",
    "shutdown",
    "production allocation",
    "contributed to oil production",
    "how much well",
    "water was injected",
    "oil was produced",
    "gas was produced",
    "oil was sold",
    "gas was flared",
    "water was transferred",
    "missing readings",
    "daily email",
    "weekly",
    "work order",
}
