"""Train the downhole dynograph classifier from labeled card CSV files.

Each CSV has one card per row: ``well`` (or ``well_key``), ``source_card_id``, ``card_date``
(or ``card_timestamp``), ``data_points`` as a JSON list of {"position", "load"}, and the label
columns ``health`` (Healthy / Watch / Problem / Bad card) and ``label`` (the detailed pattern).

    umai-train-card-classifier --train-file data/dynocard_training_data.csv \
        --train-file data/dynocard_synthetic_data.csv --output models/rod_pump_downhole_classifier.joblib

Cross-validation holds out whole wells, so the printed report reflects wells the model has
never seen. The saved file records the scikit-learn version; UMAI refuses to load it with
any other version.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np

from umai.rod_pump_health.classifier import FORMAT, POINTS_PER_CARD, card_features

TARGETS = ("health", "label")


def _card_date(value, utc_offset_hours):
    """Unix seconds (raw export) or an ISO date (training file) as ISO text."""
    text = "" if value is None else str(value)
    if text.isdigit():
        zone = timezone(timedelta(hours=utc_offset_hours))
        return datetime.fromtimestamp(int(text), zone).isoformat()
    return text


def load_cards(files, skip_before="2020-01-01", utc_offset_hours=-6):
    """Read the CSVs, drop repeated cards and bad controller clocks, and build features."""
    import pandas as pd

    frame = pd.concat([pd.read_csv(file, dtype=str) for file in files], ignore_index=True)
    frame = frame.rename(columns={"well_key": "well"})
    if "card_date" not in frame and "card_timestamp" in frame:
        frame["card_date"] = frame["card_timestamp"].map(lambda v: _card_date(v, utc_offset_hours))
    frame["card_id"] = frame["well"] + "|" + frame["source_card_id"]
    # Later files win, so a card in a field-events file replaces its label from the main file.
    frame = frame.drop_duplicates("card_id", keep="last")
    if skip_before:
        frame = frame[frame["card_date"].fillna("").str[:10] >= skip_before]
    rows, features = [], []
    for index, data_points in zip(frame.index, frame["data_points"]):
        try:
            points = json.loads(data_points)
            if len(points) < 10:
                raise ValueError("too few points")
            positions = [point["position"] for point in points]
            loads = [point["load"] for point in points]
        except (ValueError, KeyError, TypeError) as exception:
            print(f"  skipping {frame.at[index, 'card_id']}: bad data_points ({exception})")
            continue
        rows.append(index)
        features.append(card_features(positions, loads))
    return frame.loc[rows].reset_index(drop=True), np.array(features)


def train_target(frame, features, target, folds, trees, min_class_size):
    """Fit one model, printing a well-held-out cross-validation report first."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = frame[target].fillna("").values
    counts = pd.Series(labels).value_counts()
    keep = np.isin(labels, counts[counts >= min_class_size].index) & (labels != "")
    dropped = counts[(counts < min_class_size) & (counts.index != "")]
    if len(dropped):
        print(f"Leaving out classes with fewer than {min_class_size} cards: {dict(dropped)}")
    X, y, wells = features[keep], labels[keep], frame["well"].values[keep]
    print(f"{len(y)} labeled cards, target '{target}', classes: {dict(pd.Series(y).value_counts())}")
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=trees, class_weight="balanced", random_state=0, n_jobs=-1
        ),
    )
    folds = min(folds, int(pd.Series(y).value_counts().min()))
    if folds >= 2:
        cv = StratifiedGroupKFold(folds, shuffle=True, random_state=0)
        predicted = cross_val_predict(model, X, y, cv=cv, groups=wells)
        print(f"Cross-validation holds out whole wells ({folds} folds):")
        print(classification_report(y, predicted, zero_division=0))
        classes = sorted(set(y))
        print("confusion matrix (rows=true, cols=predicted):", classes)
        print(confusion_matrix(y, predicted, labels=classes))
    else:
        print("Not enough cards per class for cross-validation; skipping.")
    model.fit(X, y)
    # UMAI classifies one card per request; one core avoids starting a worker pool each time.
    model.steps[-1][1].set_params(n_jobs=1)
    return model


def main(argv=None):
    import joblib
    import sklearn

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train-file", action="append", required=True, help="labeled card CSV; repeatable")
    parser.add_argument("--output", required=True, help="model file to write")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--min-class-size", type=int, default=5)
    parser.add_argument("--skip-before", default="2020-01-01", help="ignore cards dated before this day")
    args = parser.parse_args(argv)

    frame, features = load_cards(args.train_file, args.skip_before)
    models = {}
    for target in TARGETS:
        print(f"\n==================== target: {target} ====================")
        models[target] = train_target(
            frame, features, target, args.folds, args.trees, args.min_class_size
        )
    joblib.dump(
        {
            "format": FORMAT,
            "sklearn_version": sklearn.__version__,
            "points_per_card": POINTS_PER_CARD,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "training_files": args.train_file,
            "training_cards": len(frame),
            "models": models,
        },
        args.output,
        compress=("zlib", 6),
    )
    print(f"Saved {args.output} with models for: {', '.join(TARGETS)}")


if __name__ == "__main__":
    main()
