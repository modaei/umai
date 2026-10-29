"""Surface load check: has the surface card lost its fluid load compared with the well's normal?

A tubing leak, rod part or pump that stops lifting makes the surface card collapse: the peak load
drops, the minimum rises and the load range shrinks to a fraction of normal. On the downhole card
these failures look alike, so the check watches the surface card's loads over time instead.

Each card is compared with the median of the well's own last good cards, so light and heavy wells
are treated the same. The rules are a port of the POC's surface_event_check.py. Every archived card
was pumping from 10 minutes before to 5 minutes after capture, so a card without load is a collapse,
not a stopped unit.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median

DEFAULT_SURFACE_LOAD = {
    "baseline_cards": 10,  # the well's normal is the median of its last N good cards ...
    "baseline_days": 14,  # ... from no more than this many days back
    "collapse_ratio": 0.5,  # range below this share of normal is a collapse
    "watch_ratio": 0.8,  # range below this share of normal is load dropping
    "low_load_range_lb": 3000.0,  # range below this is a collapse even without a baseline; 0 = off
}
AFTER_SHUTDOWN_HOURS = 12  # a collapse this soon after a shutdown ended may be the column refilling
EVENT_GAP_DAYS = 5  # a longer gap without cards ends a collapse
NO_LOAD_MAX_LB = 1000  # a card peaking below this never becomes part of the baseline
PROBLEM_STATUSES = ("collapse", "load_dropping")


def card_statuses(cards, limits=None):
    """Walk the cards in time order and give each a status against the well's normal at that time.

    ``cards`` are dicts with ``time`` (Unix seconds), ``max_load`` and ``min_load``. Collapsed cards
    are kept out of the baseline, so a long collapse does not become the new normal.
    """
    limits = {**DEFAULT_SURFACE_LOAD, **(limits or {})}
    window = limits["baseline_days"] * 86400
    good = []
    output = []
    for card in cards:
        time = card["time"]
        good = [entry for entry in good if time - entry["time"] <= window][-limits["baseline_cards"]:]
        normal = None
        if len(good) >= 3:
            normal = {
                key: median(entry[key] for entry in good)
                for key in ("max_load", "min_load", "load_range")
            }
        load_range = card["max_load"] - card["min_load"]
        ratio = load_range / normal["load_range"] if normal and normal["load_range"] > 0 else None
        if (ratio is not None and ratio < limits["collapse_ratio"]) or (
            limits["low_load_range_lb"] and load_range < limits["low_load_range_lb"]
        ):
            status = "collapse"
        elif ratio is not None and ratio < limits["watch_ratio"]:
            status = "load_dropping"
        elif ratio is None:
            status = "no_baseline"
        else:
            status = "normal"
        row = {**card, "load_range": load_range, "normal": normal, "range_ratio": ratio, "status": status}
        if status != "collapse" and card["max_load"] >= NO_LOAD_MAX_LB:
            good.append(row)
        output.append(row)
    return output


def surface_load_check(history, card, timezone, limits=None, shutdowns=()):
    """Status of ``card`` against the earlier ``history`` cards, with collapse details when it has one.

    ``shutdowns`` are dicts with ``end`` (aware datetime) and ``downtime_code``; one that ended
    shortly before a collapse started marks it as possibly the fluid column refilling.
    """
    rows = card_statuses([*history, card], limits)
    current = rows[-1]
    normal = current["normal"]
    result = {
        "status": current["status"],
        "card_at": _iso(current["time"], timezone),
        "max_load": round(current["max_load"]),
        "min_load": round(current["min_load"]),
        "load_range": round(current["load_range"]),
        "normal": {key: round(value) for key, value in normal.items()} if normal else None,
        "range_ratio": round(current["range_ratio"], 3) if current["range_ratio"] is not None else None,
    }
    if current["status"] not in PROBLEM_STATUSES:
        return result

    # The collapse is the run of problem cards up to this one, ended by a good card or a long gap.
    start = len(rows) - 1
    while (
        start > 0
        and rows[start - 1]["status"] in PROBLEM_STATUSES
        and rows[start]["time"] - rows[start - 1]["time"] <= EVENT_GAP_DAYS * 86400
    ):
        start -= 1
    streak = rows[start:]
    previous = next(
        (row for row in reversed(rows[:start]) if row["status"] in ("normal", "no_baseline")), None
    )
    since = datetime.fromtimestamp(streak[0]["time"], timezone)
    ranges = [row["load_range"] for row in streak]
    recovering = len(ranges) >= 3 and ranges[-1] > ranges[0] * 1.3
    shutdown = next(
        (
            entry
            for entry in sorted(shutdowns, key=lambda entry: entry["end"], reverse=True)
            if since - timedelta(hours=AFTER_SHUTDOWN_HOURS) <= entry["end"] <= since
        ),
        None,
    )
    level = None
    if normal and normal["load_range"] > 0:
        middle = (current["max_load"] + current["min_load"]) / 2
        level = round((middle - normal["min_load"]) / normal["load_range"] * 100)
    if shutdown:
        hint = (
            "Started right after a shutdown: may be the fluid column refilling after the restart. "
            "Worry if it does not recover."
        )
    elif recovering:
        hint = "Fluid load was lost and is coming back on its own: watch."
    elif current["status"] == "collapse":
        hint = (
            "Fluid load lost while pumping: check for a tubing leak, a rod part or a pump that is "
            "not lifting (look at the downhole card)."
        )
    else:
        hint = "Load range is falling below the well's normal: watch for a collapse."
    window = (limits or {}).get("baseline_days", DEFAULT_SURFACE_LOAD["baseline_days"]) * 86400
    return {
        **result,
        "since": since.isoformat(),
        "cards": len(streak),
        "last_normal_card_at": _iso(previous["time"], timezone) if previous else None,
        "normal_card_id": previous.get("id") if previous else None,
        "card_level_pct": level,
        "recovering": recovering,
        "after_shutdown": (
            {"end": shutdown["end"].isoformat(), "downtime_code": shutdown.get("downtime_code")}
            if shutdown
            else None
        ),
        "hint": hint,
        # Peak and minimum load of every card in the baseline window, for the trend chart.
        "trend": [
            [row["time"], round(row["max_load"]), round(row["min_load"])]
            for row in rows
            if current["time"] - row["time"] <= window
        ],
    }


def _iso(timestamp, timezone):
    return datetime.fromtimestamp(timestamp, timezone).isoformat()
