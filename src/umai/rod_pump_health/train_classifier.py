"""Train candidate downhole dynograph classifiers from the card labels in the archive database.

Labels come from ``rod_pump_card_labels`` (written by UMetriqs): imported rule-based and
synthetic cards, LLM reviews, human reviews and field-confirmed events. A run trains a
*candidate*, evaluates it with whole wells held out, compares it with the live model on cards
the live model never trained on, and writes a report with a recommendation. It never changes
the live model; ``umai-card-classifier promote`` does that.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import text

from umai.rod_pump_health.classifier import FORMAT, POINTS_PER_CARD, card_features
from umai.rod_pump_health.labels import ALIASES, PATTERNS, UNTRAINED_PATTERNS

TARGETS = ("health", "label")
# Which label wins when one card has several, and how much each source counts in training.
SOURCE_RANK = {"field_event": 5, "human_review": 4, "llm_review": 3, "rules": 2, "synthetic": 1}
SOURCE_WEIGHT = {"field_event": 3.0, "human_review": 3.0, "llm_review": 1.5, "rules": 1.0, "synthetic": 0.5}
# Sources confirmed outside the classifier and the LLM; the fairest yardstick for both models.
GOLD_SOURCES = frozenset({"field_event", "human_review"})

DEFAULTS = {
    "trees": 300,
    "folds": 5,
    "min_class_size": 5,
    # A label may skip the LLM only at this precision, on at least this many real cards.
    "precision_target": 0.95,
    "min_real_cards": 30,
    "min_covered_cards": 20,
    # A trusted label regresses when the candidate's precision falls this far below the live model's.
    "regression_margin": 0.02,
    "min_comparison_cards": 10,
    "disagreement_days": 30,
    # When --if-needed trains: enough new labels overall, for one untrusted label, any new gold
    # label, or the last training being this old.
    "retrain_new_labels": 200,
    "retrain_new_label_cards": 20,
    "retrain_max_age_days": 30,
}

STATUS_COMMAND = "umai-card-classifier status"


def read_labels(archive_engine):
    """All usable label rows; excluded rows are left out."""
    import pandas as pd

    with archive_engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT id, well_key, source_card_id, card_timestamp, downhole_points, health, label, "
                "label_confidence, label_source, source_detail, classifier_label, "
                "classifier_confidence, classifier_model_version, created_at "
                "FROM rod_pump_card_labels WHERE excluded = 0"
            )
        ).mappings().all()
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        return frame
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    return frame


def _decoded(value):
    if isinstance(value, (str, bytes)):
        try:
            return json.loads(value)
        except ValueError:
            return None
    return value


def prepare(frame):
    """Clean labels into one training row per card, thinned to one per well, label and day.

    LLM labels count only at high confidence; aliases map old label names; when a card has
    labels from several sources the most trusted source wins.
    """
    import pandas as pd

    if frame.empty:
        return frame, np.empty((0, 2 * POINTS_PER_CARD + 5))
    frame = frame.copy()
    frame["label"] = frame["label"].map(lambda label: ALIASES.get(label, label))
    frame = frame[~frame["label"].isin(UNTRAINED_PATTERNS)]
    frame = frame[(frame["label_source"] != "llm_review") | (frame["label_confidence"] == "high")]
    frame["health"] = [
        health if isinstance(health, str) and health else PATTERNS.get(label, ("", ""))[0]
        for health, label in zip(frame["health"], frame["label"])
    ]
    frame["source_detail"] = frame["source_detail"].map(_decoded)
    # Imported rows are the data earlier models trained on, so they never judge those models.
    frame["imported"] = frame["source_detail"].map(lambda detail: isinstance(detail, dict) and "import" in detail)
    frame["rank"] = frame["label_source"].map(SOURCE_RANK).fillna(0)
    frame["card_id"] = (
        frame["well_key"].astype(str) + "|" + frame["source_card_id"].astype(str)
        + "|" + frame["card_timestamp"].astype(str)
    )
    frame = frame.sort_values(["rank", "created_at"], ascending=False).drop_duplicates("card_id")
    # Consecutive cards of one well are nearly identical; one per well, label and day is enough.
    day = pd.to_datetime(frame["card_timestamp"].astype("int64"), unit="s").dt.date.astype(str)
    synthetic = frame["label_source"] == "synthetic"
    frame = frame[synthetic | ~frame.assign(day=day).duplicated(["well_key", "label", "day"])]
    rows, features = [], []
    for index, points in zip(frame.index, frame["downhole_points"]):
        points = _decoded(points)
        try:
            if not isinstance(points, list) or len(points) < 10:
                raise ValueError
            features.append(card_features([p["position"] for p in points], [p["load"] for p in points]))
            rows.append(index)
        except (ValueError, KeyError, TypeError):
            continue
    frame = frame.loc[rows].reset_index(drop=True)
    frame["points"] = [_decoded(points) for points in frame["downhole_points"]]
    frame["real"] = frame["label_source"] != "synthetic"
    frame["gold"] = frame["label_source"].isin(GOLD_SOURCES)
    frame["weight"] = frame["label_source"].map(SOURCE_WEIGHT).fillna(1.0)
    return frame, np.array(features).reshape(len(rows), -1)


def _pipeline(trees):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=trees, class_weight="balanced", random_state=0, n_jobs=-1),
    )


def train_target(frame, features, target, options):
    """Fit one target and predict every kept card with its well held out.

    Returns the fitted model and, per row, the held-out prediction and confidence (empty for
    rows of classes too small to train).
    """
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    labels = frame[target].fillna("").astype(str).values
    counts = pd.Series(labels).value_counts()
    keep = np.isin(labels, counts[counts >= options["min_class_size"]].index) & (labels != "")
    X, y = features[keep], labels[keep]
    weights, wells = frame["weight"].values[keep], frame["well_key"].values[keep]
    if len(set(y)) < 2:
        raise ValueError(f"Target {target} needs at least two classes with {options['min_class_size']} cards.")
    model = _pipeline(options["trees"])
    predicted = np.full(len(frame), "", dtype=object)
    confidence = np.zeros(len(frame))
    folds = min(options["folds"], int(pd.Series(y).value_counts().min()), len(set(wells)))
    if folds >= 2:
        probabilities = cross_val_predict(
            model, X, y, groups=wells, cv=StratifiedGroupKFold(folds, shuffle=True, random_state=0),
            method="predict_proba", params={"randomforestclassifier__sample_weight": weights},
        )
        classes = np.unique(y)
        predicted[keep] = classes[probabilities.argmax(1)]
        confidence[keep] = probabilities.max(1)
    model.fit(X, y, randomforestclassifier__sample_weight=weights)
    # UMAI classifies one card per request; one core avoids starting a worker pool each time.
    model.steps[-1][1].set_params(n_jobs=1)
    return model, predicted, confidence


def _precision(reference, predicted, confidence, label, threshold):
    """Precision and count of cards predicted as label at or above the threshold."""
    chosen = (predicted == label) & (confidence >= threshold)
    covered = int(chosen.sum())
    return (float((reference[chosen] == label).mean()) if covered else None), covered


def label_thresholds(frame, options, floor):
    """Per label: real cards, and the lowest threshold (not below the floor) whose held-out
    precision reaches the target on enough cards. A label qualifies to skip the LLM when both hold."""
    real = frame[frame["real"] & (frame["cv_label"] != "")]
    reference, predicted = real["label"].values, real["cv_label"].values
    confidence = real["cv_confidence"].values
    result = {}
    for label in sorted(frame["label"].unique()):
        threshold, precision, covered = None, None, 0
        for step in np.arange(floor, 1.0, 0.01):
            precision, covered = _precision(reference, predicted, confidence, label, round(step, 2))
            if covered < options["min_covered_cards"]:
                break
            if precision >= options["precision_target"]:
                threshold = round(float(step), 2)
                break
        cards = int((frame["real"] & (frame["label"] == label)).sum())
        result[label] = {
            "cards_real": cards,
            "cards_synthetic": int(((~frame["real"]) & (frame["label"] == label)).sum()),
            "cards_gold": int((frame["gold"] & (frame["label"] == label)).sum()),
            "threshold": threshold,
            "precision": None if threshold is None else round(precision, 3),
            "covered": covered if threshold is not None else 0,
            "qualifies": threshold is not None and cards >= options["min_real_cards"],
        }
    return result


def _scores(reference, predicted, confidence, trusted):
    """Accuracy, and precision of each trusted label at its threshold."""
    scores = {"cards": int(len(reference)), "accuracy": None, "trusted": {}}
    if len(reference):
        scores["accuracy"] = round(float((reference == predicted).mean()), 3)
    for label, threshold in trusted.items():
        precision, covered = _precision(reference, predicted, confidence, label, threshold)
        scores["trusted"][label] = {
            "threshold": threshold,
            "precision": None if precision is None else round(precision, 3),
            "covered": covered,
        }
    return scores


def compare(frame, live, live_trained_at, candidate_trusted):
    """Score the candidate (held-out predictions) and the live model on cards the live model
    never trained on: gold labels and labels added after it was trained, never imported rows."""
    import pandas as pd

    newer = frame["created_at"] > pd.Timestamp(live_trained_at) if live_trained_at else frame["real"]
    rows = frame[frame["real"] & ~frame["imported"] & (frame["gold"] | newer) & (frame["cv_label"] != "")]
    reference = rows["label"].values
    result = {
        "basis": "gold labels and labels added after the live model was trained",
        "candidate": _scores(reference, rows["cv_label"].values, rows["cv_confidence"].values, candidate_trusted),
        "live": None,
    }
    if live is not None and len(rows):
        predictions = [live.classify(points) for points in rows["points"]]
        live_label = np.array([p["label"] for p in predictions], dtype=object)
        live_confidence = np.array([min(p["label_confidence"], p["health_confidence"]) for p in predictions])
        result["live"] = _scores(reference, live_label, live_confidence, live.trusted or {})
        result["candidate_on_live_thresholds"] = _scores(
            reference, rows["cv_label"].values, rows["cv_confidence"].values, live.trusted or {}
        )
        frame.loc[rows.index, "live_label"] = live_label
        frame.loc[rows.index, "live_confidence"] = live_confidence
    return result, rows.index


def disagreements(frame, live, days, now):
    """How often the LLM disagreed with confident classifier calls of the live model, per label.
    These cards went to the LLM anyway (audit sample, stale review or untrusted label)."""
    import pandas as pd

    if live is None:
        return {}
    since = pd.Timestamp(now - timedelta(days=days))
    rows = frame[
        (frame["label_source"] == "llm_review") & (frame["created_at"] >= since)
        & (frame["classifier_model_version"] == live.version) & frame["classifier_label"].notna()
    ]
    result = {}
    for label, threshold in (live.trusted or {}).items():
        chosen = rows[(rows["classifier_label"] == label) & (rows["classifier_confidence"].astype(float) >= threshold)]
        if len(chosen):
            agreed = int((chosen["label"] == label).sum())
            result[label] = {"reviews": int(len(chosen)), "agreed": agreed, "precision": round(agreed / len(chosen), 3)}
    return result


def recommend(version, labels, comparison, live_state, disagreed, options):
    """Decide whether to promote, which new labels to trust, and which live labels to untrust."""
    live_trusted = live_state["trusted"]
    reasons, untrust = [], []
    keep = [label for label in live_trusted if labels.get(label, {}).get("qualifies")]
    lose = [label for label in live_trusted if label not in keep]
    new = [label for label, entry in labels.items() if entry["qualifies"] and label not in live_trusted]
    for label in lose:
        entry = labels.get(label)
        if entry is None:
            reasons.append(f"{label} is trusted now but has no training cards, so the candidate cannot decide it.")
        elif entry["threshold"] is not None:
            reasons.append(
                f"{label} has only {entry['cards_real']} real cards; {options['min_real_cards']} are needed to trust it."
            )
        else:
            reasons.append(
                f"{label} no longer reaches {options['precision_target']:.0%} precision in the candidate."
            )
    regressions = []
    live_scores = comparison.get("live")
    candidate_scores = comparison.get("candidate_on_live_thresholds")
    if live_scores and candidate_scores:
        if (live_scores["accuracy"] is not None and candidate_scores["accuracy"] is not None
                and live_scores["cards"] >= options["min_comparison_cards"]
                and candidate_scores["accuracy"] < live_scores["accuracy"] - options["regression_margin"]):
            regressions.append(
                f"accuracy fell from {live_scores['accuracy']:.0%} to {candidate_scores['accuracy']:.0%}"
            )
        for label, score in live_scores["trusted"].items():
            ours = candidate_scores["trusted"].get(label, {})
            if (score["covered"] >= options["min_comparison_cards"] and score["precision"] is not None
                    and ours.get("precision") is not None
                    and ours["precision"] < score["precision"] - options["regression_margin"]):
                regressions.append(
                    f"{label} precision fell from {score['precision']:.0%} to {ours['precision']:.0%}"
                )
            if (score["covered"] >= options["min_comparison_cards"] and score["precision"] is not None
                    and score["precision"] < options["precision_target"]):
                untrust.append({"label": label, "reason": (
                    f"the live model's {label} calls were right {score['precision']:.0%} of the time "
                    f"on {score['covered']} newer cards"
                )})
    for label, entry in disagreed.items():
        if (entry["reviews"] >= options["min_comparison_cards"] and entry["precision"] < options["precision_target"]
                and label not in {item["label"] for item in untrust}):
            untrust.append({"label": label, "reason": (
                f"the LLM agreed with {entry['agreed']} of {entry['reviews']} confident {label} calls "
                f"in the last {options['disagreement_days']} days"
            )})
    reasons.extend(f"Compared with the live model, {item}." for item in regressions)
    if live_state["source"] == "none":
        action = "promote_with_new_labels" if new else "promote"
        reasons.append("There is no live model yet.")
    elif lose or regressions:
        action = "keep_current"
    else:
        action = "promote_with_new_labels" if new else "promote"
        if not live_scores:
            reasons.append("No cards newer than the live model to compare on yet; judged on held-out wells only.")
    promote = f"umai-card-classifier promote {version}"
    commands = {"status": STATUS_COMMAND, "untrust": [f"umai-card-classifier untrust {item['label']}" for item in untrust]}
    if action != "keep_current":
        commands["promote"] = promote + "".join(f" --trust {label}" for label in new)
        if new:
            commands["promote_without_new_labels"] = promote
    return {
        "action": action,
        "reasons": reasons,
        "trusted_kept": keep,
        "trusted_lost": lose,
        "new_trusted": new,
        "untrust": untrust,
        "commands": commands,
    }


def examples(frame, comparison_rows, labels, live_state, limit=10):
    """Cards worth a look: confident candidate mistakes on the comparison cards, then one card
    of each newly qualifying label."""
    chosen = []
    rows = frame.loc[comparison_rows]
    wrong = rows[rows["cv_label"] != rows["label"]].sort_values("cv_confidence", ascending=False)
    chosen.extend(wrong.index[: limit // 2 or 1])
    for label, entry in labels.items():
        if entry["qualifies"] and label not in live_state["trusted"] and len(chosen) < limit:
            match = frame[frame["real"] & (frame["label"] == label) & (frame["cv_label"] == label)]
            if len(match):
                chosen.append(match["cv_confidence"].idxmax())
    result = []
    for index in chosen[:limit]:
        row = frame.loc[index]
        points = row["points"]
        step = max(1, len(points) // 100)
        result.append({
            "kind": "new_label" if row["cv_label"] == row["label"] else "disagreement",
            "well_key": row["well_key"],
            "card_date": datetime.fromtimestamp(int(row["card_timestamp"]), timezone.utc).isoformat(),
            "reference_label": row["label"],
            "reference_source": row["label_source"],
            "candidate_label": row["cv_label"],
            "candidate_confidence": round(float(row["cv_confidence"]), 3),
            "live_label": row.get("live_label") if isinstance(row.get("live_label"), str) else None,
            "points": [[round(p["position"], 2), round(p["load"])] for p in points[::step]],
        })
    return result


def data_summary(frame, raw, live_trained_at):
    """Label counts by source and label, and what is new since the live model was trained."""
    import pandas as pd

    summary = {
        "label_rows": int(len(raw)),
        "training_cards": int(len(frame)),
        "by_source": {k: int(v) for k, v in frame["label_source"].value_counts().items()},
        "new_since_live": None,
    }
    if live_trained_at and len(raw):
        new = raw[raw["created_at"] > pd.Timestamp(live_trained_at)]
        summary["new_since_live"] = {
            "total": int(len(new)),
            "by_source": {k: int(v) for k, v in new["label_source"].value_counts().items()},
            "by_label": {k: int(v) for k, v in new["label"].value_counts().items()},
        }
    return summary


def weekly_llm_savings(frame, labels, days, now):
    """Rough LLM calls a week each newly qualifying label would save: recent LLM-labelled cards
    the candidate would have decided itself."""
    import pandas as pd

    since = pd.Timestamp(now - timedelta(days=days))
    recent = frame[(frame["label_source"] == "llm_review") & (frame["created_at"] >= since)]
    for label, entry in labels.items():
        if entry["threshold"] is not None:
            count = int(((recent["cv_label"] == label) & (recent["cv_confidence"] >= entry["threshold"])).sum())
            entry["weekly_llm_calls_saved"] = round(count * 7 / days, 1)


def last_trained_at(registry):
    """When the newest candidate was trained, from its report."""
    versions = registry.candidates()
    report = registry.report(versions[-1]) if versions else None
    return report.get("trained_at") if report else None


def should_train(raw, registry, options, now):
    """Whether --if-needed should train now, and why."""
    import pandas as pd

    trained_at = last_trained_at(registry)
    if not trained_at:
        return True, "no earlier training run"
    trained = pd.Timestamp(trained_at)
    if now - trained.to_pydatetime() >= timedelta(days=options["retrain_max_age_days"]):
        return True, f"last training is over {options['retrain_max_age_days']} days old"
    if raw.empty:
        return False, "no labels"
    new = raw[raw["created_at"] > trained]
    if len(new) >= options["retrain_new_labels"]:
        return True, f"{len(new)} new labels"
    if new["label_source"].isin(GOLD_SOURCES).any():
        return True, "new field-confirmed or human-reviewed labels"
    trusted = registry.state()["trusted"]
    counts = new[~new["label"].isin(trusted)]["label"].value_counts()
    if len(counts) and counts.iloc[0] >= options["retrain_new_label_cards"]:
        return True, f"{counts.iloc[0]} new {counts.index[0]} labels"
    return False, f"only {len(new)} new labels since {trained_at}"


def train(archive_engine, registry, floor, options=None, now=None, wells_reviewed=None):
    """Train, evaluate and save one candidate; returns its report (the live model is unchanged)."""
    import sklearn

    options = {**DEFAULTS, **(options or {})}
    now = now or datetime.now(timezone.utc)
    version = now.strftime("%Y%m%d-%H%M%S")
    raw = read_labels(archive_engine)
    frame, features = prepare(raw)
    if frame.empty:
        raise ValueError("No usable card labels in rod_pump_card_labels.")
    models = {}
    for target in TARGETS:
        models[target], predicted, confidence = train_target(frame, features, target, options)
        frame[f"cv_{target}"], frame[f"cv_{target}_confidence"] = predicted, confidence
    frame["cv_confidence"] = np.minimum(frame["cv_health_confidence"], frame["cv_label_confidence"])
    frame["live_label"], frame["live_confidence"] = None, np.nan
    live = registry.active()
    live_state = registry.state()
    live_report = registry.report(live_state["version"]) if live_state["source"] == "registry" else None
    live_trained_at = (live_report or {}).get("trained_at") or _model_trained_at(registry, live_state)
    labels = label_thresholds(frame, options, floor)
    weekly_llm_savings(frame, labels, options["disagreement_days"], now)
    candidate_trusted = {label: entry["threshold"] for label, entry in labels.items() if entry["qualifies"]}
    comparison, comparison_rows = compare(frame, live, live_trained_at, candidate_trusted)
    disagreed = disagreements(frame, live, options["disagreement_days"], now)
    recommendation = recommend(version, labels, comparison, live_state, disagreed, options)
    report = {
        "version": version,
        "trained_at": now.isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "settings": {**options, "min_confidence": floor},
        "live": {
            "source": live_state["source"],
            "version": live_state["version"],
            "trusted": live_state["trusted"],
            "trained_at": live_trained_at,
        },
        "data": data_summary(frame, raw, live_trained_at),
        "labels": labels,
        "comparison": comparison,
        "disagreements": disagreed,
        "recommendation": recommendation,
        # Promoting changes the classifier version, so every well's next card goes to the LLM once.
        "re_review_wells": wells_reviewed,
        "examples": examples(frame, comparison_rows, labels, live_state),
    }
    registry.save_candidate(
        version,
        {
            "format": FORMAT,
            "sklearn_version": sklearn.__version__,
            "points_per_card": POINTS_PER_CARD,
            "trained_at": report["trained_at"],
            "training_cards": len(frame),
            "models": models,
        },
        report,
    )
    return report


def _model_trained_at(registry, state):
    """Training time saved inside the live model file, for models without a report."""
    import joblib

    if state["source"] == "none":
        return None
    path = registry.model_path(state["version"])
    path = path if path.is_file() else registry.legacy_model
    try:
        return joblib.load(path).get("trained_at")
    except Exception:
        return None
