"""Step 0 probe: does OpenRouter's batch API honor the rod-pump health request settings?

Submits one small batch built from real cards with the same prompt, JSON schema, reasoning
effort and 60/100 point reduction as the live health check, waits for it, and reports:

  * whether every answer is plain JSON that passes the Assessment schema (json_schema enforced)
  * whether reasoning_effort is honored (same card at the configured effort and at "low")
  * per-request token usage and the batch cost
  * how long each batch stage took

Read-only against the database; writes only the raw batch result to batch_probe_<id>.json.

Run from the umai root:
    PYTHONPATH=src .venv/bin/python batch_probe.py --dry-run    # build requests, submit nothing
    PYTHONPATH=src .venv/bin/python batch_probe.py              # submit and wait
    PYTHONPATH=src .venv/bin/python batch_probe.py --batch-id ID   # resume waiting
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone

import httpx
from sqlalchemy import bindparam, text

from umai.config.settings import Settings
from umai.rod_pump_health.analysis import ASSESSMENT_SCHEMA, SYSTEM_PROMPT, Assessment, analyze
from umai.rod_pump_health.reader import RecentCardReference, analyzer_from_settings, decoded

TERMINAL = {"completed", "failed", "expired", "cancelled"}


class Captured(Exception):
    """Stops analyze() once the exact LLM content has been built."""


def sample_cards(analyzer, count):
    """Latest current card (older than 5 minutes) of the first configured wells with a downhole card."""
    with analyzer.engine.connect() as connection:
        sites = connection.execute(text(
            "SELECT site_id, related_entities FROM report_configurations "
            "WHERE function_name='rod_pump_monitoring'"
        )).mappings().all()
        well_ids = [int(i) for row in sites for i in decoded(row["related_entities"], {}).get("well_ids", [])]
        rows = connection.execute(text(
            "SELECT c.id, c.timestamp, d.id AS pull_id, d.pull_timestamp, d.well_id "
            "FROM rod_pump_cards c "
            "JOIN rod_pump_monitoring_data_cards link ON link.rod_pump_card_id=c.id "
            "JOIN rod_pump_monitoring_data d ON d.id=link.rod_pump_monitoring_data_id "
            "WHERE LOWER(c.card_type)='current' AND d.well_id IN :ids "
            "AND c.timestamp < UNIX_TIMESTAMP() - 600 "
            "AND c.id = (SELECT MAX(c2.id) FROM rod_pump_cards c2 "
            "  JOIN rod_pump_monitoring_data_cards l2 ON l2.rod_pump_card_id=c2.id "
            "  JOIN rod_pump_monitoring_data d2 ON d2.id=l2.rod_pump_monitoring_data_id "
            "  WHERE d2.well_id=d.well_id AND LOWER(c2.card_type)='current' "
            "  AND c2.timestamp < UNIX_TIMESTAMP() - 600) "
            "ORDER BY d.well_id LIMIT 20"
        ).bindparams(bindparam("ids", expanding=True)), {"ids": well_ids}).mappings().all()
    packets = []
    for row in rows:
        reference = RecentCardReference(
            source="live", id=row["id"], monitoring_data_id=row["pull_id"],
            pull_timestamp=row["pull_timestamp"], card_timestamp=row["timestamp"],
            pumping_window_start=row["timestamp"] - 600, pumping_window_end=row["timestamp"] + 300,
        )
        try:
            packet = analyzer.prepare(row["well_id"], reference)
        except ValueError as exception:
            print(f"  skip well {row['well_id']}: {exception}")
            continue
        if packet.card.downhole:
            packets.append(packet)
        if len(packets) == count:
            break
    return packets


def user_content(packet, limits):
    """The exact text the live health check sends for this card."""
    sent = {}
    # analyze() logs the deliberate stop below as a failed review; keep the output clean.
    logging.getLogger("umai.rod_pump_health.analysis").setLevel(logging.ERROR)

    def capture(content):
        sent["text"] = content[0]["text"]
        raise Captured()

    analyze(packet, capture, limits)
    return sent["text"]


def body(content, effort):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "rod_pump_health_assessment", "strict": True, "schema": ASSESSMENT_SCHEMA},
        },
        "reasoning_effort": effort,
        "max_tokens": 16000,
    }


def check(result):
    """Summarize one batch result without assuming its exact wrapping."""
    if result.get("error"):
        return {"error": result["error"]}
    response = result.get("response") or {}
    response = response.get("body", response) if isinstance(response, dict) else {}
    choice = (response.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    usage = response.get("usage") or {}
    report = {
        "finish_reason": choice.get("finish_reason"),
        "provider": response.get("provider"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "cost": usage.get("cost"),
    }
    try:
        parsed = json.loads(content)
        report["plain_json"] = True
        report["extra_keys"] = sorted(set(parsed) - set(ASSESSMENT_SCHEMA["properties"]))
        Assessment.model_validate(parsed)
        report["schema_valid"] = True
        report["health_status"] = parsed.get("health_status")
        report["findings"] = len(parsed.get("findings", []))
    except Exception as exception:
        report.setdefault("plain_json", False)
        report["schema_valid"] = False
        report["schema_error"] = f"{type(exception).__name__}: {str(exception)[:300]}"
        report["content_start"] = str(content)[:300]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="build the requests but submit nothing")
    parser.add_argument("--batch-id", help="resume waiting for an already submitted batch")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-minutes", type=int, default=90)
    args = parser.parse_args()

    settings = Settings.from_env()
    base = settings.llm_base_url.rstrip("/")
    model = settings.rod_pump_health_llm_model.removesuffix(":batch")
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    limits = {"surface": settings.rod_pump_health_surface_points, "downhole": settings.rod_pump_health_downhole_points}
    effort = settings.rod_pump_health_llm_effort
    print(f"model={model} effort={effort} limits={limits} base={base}")

    batch_id = args.batch_id
    if not batch_id:
        analyzer = analyzer_from_settings(settings)
        packets = sample_cards(analyzer, 2)
        if len(packets) < 2:
            raise SystemExit("Found fewer than 2 configured wells with a usable current card.")
        first, second = (user_content(packet, limits) for packet in packets)
        requests = [
            {"custom_id": f"well-{packets[0].well_id}", "body": body(first, effort)},
            {"custom_id": f"well-{packets[1].well_id}", "body": body(second, effort)},
            # Same card as the first request at low effort: fewer reasoning tokens means effort is honored.
            {"custom_id": f"well-{packets[0].well_id}-low", "body": body(first, "low")},
        ]
        for packet, request in zip(packets + packets[:1], requests):
            print(f"  {request['custom_id']}: card {packet.card.reference} "
                  f"at {packet.card.timestamp.isoformat()}, user content {len(request['body']['messages'][1]['content'])} chars")
        payload = {"endpoint": "/v1/chat/completions", "model": model, "requests": requests}
        if args.dry_run:
            print(f"dry run: payload {len(json.dumps(payload))} bytes, {len(requests)} requests; nothing submitted")
            return
        response = httpx.post(f"{base}/batches", json=payload, headers=headers, timeout=60)
        print(f"submit: HTTP {response.status_code}")
        if response.status_code >= 400:
            raise SystemExit(response.text[:2000])
        print(f"submit response: {response.text[:1500]}")
        batch_id = response.json()["id"]
        print(f"batch_id={batch_id}  (resume with --batch-id {batch_id})")

    started, last_status, batch, not_found = time.monotonic(), None, {}, 0
    last_state, last_print = None, 0
    while time.monotonic() - started < args.max_wait_minutes * 60:
        response = httpx.get(f"{base}/batches/{batch_id}", headers=headers, timeout=60)
        if response.status_code == 404:
            # A freshly submitted batch may not be readable straight away.
            not_found += 1
            if not_found == 1:
                print(f"status: HTTP 404 {response.text[:500]}; retrying for up to 5 minutes")
            if time.monotonic() - started < 300:
                time.sleep(15)
                continue
            listing = httpx.get(f"{base}/batches", headers=headers, timeout=60)
            print(f"still 404 after 5 min; GET /batches: HTTP {listing.status_code} {listing.text[:3000]}")
            return
        if not_found:
            print(f"batch visible after {round(time.monotonic() - started)}s")
            not_found = 0
        response.raise_for_status()
        batch = response.json()
        elapsed = round(time.monotonic() - started)
        state = (batch.get("status"), json.dumps(batch.get("request_counts"), sort_keys=True))
        # Print every change, plus a heartbeat every 5 minutes while nothing changes.
        if state != last_state or elapsed - last_print >= 300:
            last_state, last_print = state, elapsed
            print(f"[{datetime.now(timezone.utc):%H:%M:%S}Z +{elapsed}s] status={state[0]} "
                  f"counts={batch.get('request_counts')}", flush=True)
        last_status = batch.get("status")
        if last_status in TERMINAL:
            break
        time.sleep(args.poll_seconds)
    else:
        print(f"Still {last_status} after {args.max_wait_minutes} min; resume later with --batch-id {batch_id}")
        return

    path = f"batch_probe_{batch_id}.json"
    with open(path, "w") as handle:
        json.dump(batch, handle, indent=2)
    print(f"raw batch saved to {path}")
    print("batch-level:", json.dumps({key: batch.get(key) for key in (
        "status", "provider", "usage", "error", "request_counts", "created_at", "completed_at", "expires_at",
    )}, default=str))
    results = batch.get("results") or batch.get("data") or []
    if results:
        print("first result keys:", sorted(results[0]), "| response keys:",
              sorted((results[0].get("response") or {}).keys()))
    for result in sorted(results, key=lambda r: str(r.get("custom_id"))):
        print(f"{result.get('custom_id')}: {json.dumps(check(result), default=str)}")


if __name__ == "__main__":
    main()
