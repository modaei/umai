"""``umai-card-classifier``: train candidates and manage the live downhole classifier.

    umai-card-classifier train [--if-needed]      train a candidate from the archive labels
    umai-card-classifier status                   live model, trusted labels, recent changes
    umai-card-classifier promote VERSION [--trust LABEL]... [--untrust LABEL]...
    umai-card-classifier trust LABEL [--threshold T] [--force]
    umai-card-classifier untrust LABEL
    umai-card-classifier rollback

Training never changes the live model; promote, trust, untrust and rollback do, and umai picks
the change up on its next analysis.
"""

from __future__ import annotations

import argparse
import getpass
import sys
import traceback
from datetime import datetime, timezone

from sqlalchemy import text

from umai.rod_pump_health.classifier_registry import RegistryError
from umai.rod_pump_health.labels import PATTERNS


def _settings():
    from umai.config.settings import Settings

    return Settings.from_env()


def _registry(settings):
    from umai.rod_pump_health.reader import classifier_registry

    return classifier_registry(settings)


def _wells_reviewed(operational_engine):
    """Wells with a saved health review: each goes back to the LLM once after a promotion."""
    try:
        with operational_engine.connect() as connection:
            return int(connection.execute(text("SELECT COUNT(*) FROM rod_pump_health_analyses")).scalar())
    except Exception:
        return None


def train_command(args, settings, registry) -> int:
    """Train a candidate (or skip when --if-needed finds too little new data) and record the run."""
    from umai.rod_pump_health import train_classifier
    from umai.rod_pump_health.reader import database_engines

    started = datetime.now(timezone.utc)
    run = {"started_at": started.isoformat(timespec="seconds")}
    try:
        operational, archive = database_engines(settings)
        options = {"trees": args.trees, "folds": args.folds}
        if args.if_needed:
            needed, reason = train_classifier.should_train(
                train_classifier.read_labels(archive), registry, train_classifier.DEFAULTS, started
            )
            if not needed:
                registry.record_run({**run, "status": "skipped", "reason": reason, "finished_at": _now()})
                print(f"Skipped: {reason}.")
                return 0
            run["reason"] = reason
        report = train_classifier.train(
            archive, registry, settings.rod_pump_health_classifier_min_confidence, options,
            now=started, wells_reviewed=_wells_reviewed(operational),
        )
    except Exception as exception:
        registry.record_run({**run, "status": "failed", "error": f"{type(exception).__name__}: {exception}",
                             "finished_at": _now()})
        traceback.print_exc()
        return 1
    registry.record_run({**run, "status": "trained", "version": report["version"], "finished_at": _now()})
    recommendation = report["recommendation"]
    print(f"Trained candidate {report['version']}: {recommendation['action'].replace('_', ' ')}.")
    for reason in recommendation["reasons"]:
        print(f"  - {reason}")
    for name, command in recommendation["commands"].items():
        for line in command if isinstance(command, list) else [command]:
            print(f"  {name}: {line}")
    return 0


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def print_state(state, out=None):
    """Human-readable live model: version, trusted labels with thresholds, latest changes."""
    out = out or sys.stdout
    if state["source"] == "none":
        print("No live classifier: every card goes to the LLM.", file=out)
        return
    origin = "original model file, not yet in the registry" if state["source"] == "legacy" else "registry"
    print(f"Live model: {state['version']} ({origin})", file=out)
    if state.get("updated_at"):
        print(f"Last changed {state['updated_at']} by {state['updated_by']}", file=out)
    print("Trusted labels (skip the LLM at or above this confidence):", file=out)
    for label, threshold in sorted(state["trusted"].items()):
        print(f"  {label:<28} {threshold:.2f}", file=out)
    if not state["trusted"]:
        print("  none: every card goes to the LLM", file=out)
    if state["previous"]:
        print(f"Rollback goes to: {state['previous'][0]}", file=out)
    for entry in state["history"][-5:]:
        print(f"  {entry['at']}  {entry['by']}: {entry['action']}", file=out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="umai-card-classifier", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train a candidate from the archive card labels")
    train.add_argument("--if-needed", action="store_true", help="skip when there is too little new data")
    train.add_argument("--trees", type=int, default=300)
    train.add_argument("--folds", type=int, default=5)
    commands.add_parser("status", help="show the live model and its trusted labels")
    promote = commands.add_parser("promote", help="make a trained candidate the live model")
    promote.add_argument("version")
    promote.add_argument("--trust", action="append", default=[], choices=list(PATTERNS), metavar="LABEL")
    promote.add_argument("--untrust", action="append", default=[], metavar="LABEL")
    promote.add_argument("--force", action="store_true", help="trust labels the report does not qualify")
    trust = commands.add_parser("trust", help="let the live model decide a label without the LLM")
    trust.add_argument("label", choices=list(PATTERNS), metavar="LABEL")
    trust.add_argument("--threshold", type=float)
    trust.add_argument("--force", action="store_true", help="trust a label the report does not qualify")
    untrust = commands.add_parser("untrust", help="send a label back to the LLM")
    untrust.add_argument("label")
    commands.add_parser("rollback", help="return to the previous live model")
    for command in commands.choices.values():
        command.add_argument("--by", default=getpass.getuser(), help="name recorded with the change")
    args = parser.parse_args(argv)

    settings = _settings()
    registry = _registry(settings)
    try:
        if args.command == "train":
            return train_command(args, settings, registry)
        if args.command == "promote":
            state, notes = registry.promote(args.version, args.by, args.trust, args.untrust, args.force)
            for note in notes:
                print(note)
        elif args.command == "trust":
            state = registry.set_trust(args.label, True, args.by, args.threshold, args.force)
        elif args.command == "untrust":
            state = registry.set_trust(args.label, False, args.by)
        elif args.command == "rollback":
            state = registry.rollback(args.by)
        else:
            state = registry.state()
    except RegistryError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    print_state(state)
    return 0


def latest_training(registry) -> dict:
    """The last training run and, when it trained a candidate, that candidate's report.

    latest_candidate is the newest trained candidate, so an email about a skipped or failed run
    can still say whether an earlier candidate is waiting to be promoted.
    """
    run = registry.latest_run()
    report = registry.report(run.get("version")) if run else None
    candidates = registry.candidates()
    latest = registry.report(candidates[-1]) if candidates else None
    return {"run": run, "report": report, "latest_candidate": latest, "live": registry.state()}


if __name__ == "__main__":
    sys.exit(main())
