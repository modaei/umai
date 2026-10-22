import json

from umai.services.rag_sources import extract_rag_sources, format_rag_source


def rag_trace(matches, ok=True):
    return {
        "tool": "search_operational_context",
        "arguments": {"query": "what happened"},
        "result": json.dumps({"ok": ok, "matches": matches}),
    }


def match(chunk_id="chunk-1", text="A useful operational note."):
    return {
        "chunk_id": chunk_id,
        "source_type": "well_shutdown",
        "event_date": "2026-06-14",
        "entity_name": "HARTZOG DRAW UNIT 5144H",
        "text": text,
    }


def test_extract_rag_sources_from_operational_context_trace():
    sources = extract_rag_sources([rag_trace([match()])])

    assert sources == [
        {
            "chunk_id": "chunk-1",
            "source_type": "well_shutdown",
            "event_date": "2026-06-14",
            "entity_name": "HARTZOG DRAW UNIT 5144H",
            "text": "A useful operational note.",
        }
    ]


def test_extract_rag_sources_ignores_non_rag_and_failed_results():
    sources = extract_rag_sources(
        [
            {"tool": "run_report", "result": json.dumps({"matches": [match()]})},
            rag_trace([match()], ok=False),
        ]
    )

    assert sources == []


def test_extract_rag_sources_deduplicates_by_chunk_id():
    sources = extract_rag_sources([rag_trace([match(), match()])])

    assert len(sources) == 1


def test_extract_rag_sources_limits_results_to_ten():
    matches = [match(chunk_id=f"chunk-{index}") for index in range(12)]

    sources = extract_rag_sources([rag_trace(matches)])

    assert len(sources) == 10
    assert sources[-1]["chunk_id"] == "chunk-9"


def test_format_rag_source_uses_metadata_and_preview_text():
    rendered = format_rag_source(match())

    assert "well_shutdown" in rendered
    assert "2026-06-14" in rendered
    assert "HARTZOG DRAW UNIT 5144H" in rendered
    assert "A useful operational note." in rendered
