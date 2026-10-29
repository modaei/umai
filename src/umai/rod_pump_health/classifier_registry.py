"""Versioned downhole classifier models, the live model pointer and its trusted labels.

The registry folder holds ``<version>.joblib`` and ``<version>.report.json`` for every trained
candidate, ``current.json`` naming the live model and the labels it may decide without the LLM,
and ``runs/latest.json`` describing the last training run. Training only adds candidates; the
live model and its trusted labels change only through the ``umai-card-classifier`` commands.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from umai.rod_pump_health.classifier import DownholeClassifier, file_version, load_classifier
from umai.rod_pump_health.labels import LEGACY_TRUSTED_LABELS

POINTER = "current.json"
LATEST_RUN = "runs/latest.json"
# Candidate models kept besides the live one and the versions rollback can return to.
KEEP_CANDIDATES = 5
# Earlier live models rollback can return to.
KEEP_PREVIOUS = 5


class RegistryError(ValueError):
    """A registry command that cannot be carried out as asked."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, data) -> None:
    """Replace the file atomically so umai never reads a half-written pointer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str))
    os.replace(temporary, path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class ClassifierRegistry:
    """File-based model registry. Without ``current.json`` the legacy model file stays live with
    the legacy trusted labels at the minimum confidence, exactly as before the registry."""

    def __init__(self, directory: str, legacy_model: str = "", min_confidence: float = 0.9):
        self.directory = Path(directory)
        self.legacy_model = legacy_model
        self.min_confidence = min_confidence
        self._active_key = None
        self._active = None

    def model_path(self, version: str) -> Path:
        return self.directory / f"{version}.joblib"

    def report_path(self, version: str) -> Path:
        return self.directory / f"{version}.report.json"

    def report(self, version: str | None) -> dict | None:
        return _read_json(self.report_path(version)) if version else None

    def pointer(self) -> dict | None:
        return _read_json(self.directory / POINTER)

    def state(self) -> dict:
        """The live model: version, trusted labels with thresholds, and where it comes from."""
        pointer = self.pointer()
        if pointer and pointer.get("version"):
            return {
                "source": "registry",
                "version": pointer["version"],
                "trusted": dict(pointer.get("trusted") or {}),
                "updated_at": pointer.get("updated_at"),
                "updated_by": pointer.get("updated_by"),
                "previous": [entry["version"] for entry in pointer.get("previous", [])],
                "history": pointer.get("history", []),
            }
        if self.legacy_model and Path(self.legacy_model).is_file():
            return {
                "source": "legacy",
                "version": file_version(self.legacy_model),
                "trusted": {label: self.min_confidence for label in sorted(LEGACY_TRUSTED_LABELS)},
                "updated_at": None,
                "updated_by": None,
                "previous": [],
                "history": [],
            }
        return {"source": "none", "version": None, "trusted": {}, "previous": [], "history": []}

    def active(self) -> DownholeClassifier | None:
        """The live classifier with its trusted labels, reloaded when the pointer file changes."""
        pointer = self.directory / POINTER
        try:
            key = ("registry", pointer.stat().st_mtime_ns)
        except OSError:
            key = ("legacy", self.legacy_model)
        if key != self._active_key:
            state = self.state()
            if state["source"] == "registry":
                path = str(self.model_path(state["version"]))
            elif state["source"] == "legacy":
                path = self.legacy_model
            else:
                path = ""
            loaded = load_classifier(path, state["version"]) if path else None
            self._active = (
                DownholeClassifier(loaded.models, loaded.version, state["trusted"]) if loaded else None
            )
            self._active_key = key
        return self._active

    def candidates(self) -> list[str]:
        """Trained versions with both a model file and a report, oldest first."""
        return sorted(
            path.name[: -len(".report.json")]
            for path in self.directory.glob("*.report.json")
            if self.model_path(path.name[: -len(".report.json")]).is_file()
        )

    def latest_run(self) -> dict | None:
        return _read_json(self.directory / LATEST_RUN)

    def record_run(self, run: dict) -> None:
        _write_json(self.directory / LATEST_RUN, run)

    def save_candidate(self, version: str, saved_model: dict, report: dict) -> None:
        """Write a trained candidate; the live model is not touched."""
        import joblib

        self.directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(saved_model, self.model_path(version), compress=("zlib", 6))
        _write_json(self.report_path(version), report)
        self.prune()

    def prune(self) -> None:
        """Drop old candidates, keeping the newest few, the live model and rollback targets."""
        state = self.state()
        keep = set(self.candidates()[-KEEP_CANDIDATES:]) | {state["version"]} | set(state["previous"])
        for version in self.candidates():
            if version not in keep:
                self.model_path(version).unlink(missing_ok=True)
                self.report_path(version).unlink(missing_ok=True)

    def _pointer_for_change(self) -> dict:
        """The pointer to modify; the legacy model is first copied into the registry under its
        own hash, so its version, and with it every saved review, stays valid."""
        pointer = self.pointer()
        if pointer and pointer.get("version"):
            return pointer
        state = self.state()
        if state["source"] != "legacy":
            raise RegistryError("There is no live model yet; promote a trained version first.")
        self.directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.legacy_model, self.model_path(state["version"]))
        return {
            "version": state["version"],
            "trusted": state["trusted"],
            "previous": [],
            "history": [{"at": _now(), "by": "registry", "action": "adopt legacy model",
                         "version": state["version"], "trusted": state["trusted"]}],
        }

    def _save_pointer(self, pointer: dict, by: str, action: str) -> dict:
        pointer["updated_at"] = _now()
        pointer["updated_by"] = by
        pointer["history"] = (pointer.get("history") or []) + [
            {"at": pointer["updated_at"], "by": by, "action": action,
             "version": pointer["version"], "trusted": pointer["trusted"]}
        ]
        _write_json(self.directory / POINTER, pointer)
        return self.state()

    def _qualifying(self, version: str) -> dict[str, float]:
        """Labels the version's training report found reliable enough, with their thresholds."""
        report = self.report(version) or {}
        return {
            label: entry["threshold"]
            for label, entry in (report.get("labels") or {}).items()
            if entry.get("qualifies") and entry.get("threshold") is not None
        }

    def promote(self, version: str, by: str, trust=(), untrust=(), force=False) -> tuple[dict, list[str]]:
        """Make a trained version live. Labels trusted now stay trusted when the version's report
        still qualifies them (at its thresholds); new labels are trusted only when named.
        Returns the new state and notes on labels that were dropped."""
        if not self.model_path(version).is_file():
            raise RegistryError(f"No trained model {version} in {self.directory}.")
        if load_classifier(str(self.model_path(version)), version) is None:
            raise RegistryError(f"Model {version} cannot be loaded; see the umai log.")
        qualifying = self._qualifying(version)
        state = self.state()
        trusted, notes = {}, []
        for label in state["trusted"]:
            if label in untrust:
                continue
            if label in qualifying:
                trusted[label] = qualifying[label]
            else:
                notes.append(f"{label} is no longer trusted: model {version} does not qualify it.")
        for label in trust:
            if label in qualifying:
                trusted[label] = qualifying[label]
            elif force:
                trusted[label] = self.min_confidence
            else:
                raise RegistryError(
                    f"Model {version} does not qualify {label}; use --force to trust it anyway."
                )
        pointer = self._pointer_for_change() if state["source"] != "none" else {"previous": []}
        if pointer.get("version") and pointer["version"] != version:
            pointer["previous"] = (
                [{"version": pointer["version"], "trusted": pointer["trusted"]}]
                + pointer.get("previous", [])
            )[:KEEP_PREVIOUS]
        pointer["version"] = version
        pointer["trusted"] = trusted
        return self._save_pointer(pointer, by, f"promote {version}"), notes

    def set_trust(self, label: str, trusted: bool, by: str, threshold=None, force=False) -> dict:
        """Trust or untrust one label of the live model without changing the model, so saved
        reviews stay valid and only that label's routing changes."""
        pointer = self._pointer_for_change()
        labels = dict(pointer["trusted"])
        if not trusted:
            if label not in labels:
                raise RegistryError(f"{label} is not trusted.")
            labels.pop(label)
        else:
            qualifying = self._qualifying(pointer["version"])
            if threshold is None and label in qualifying:
                threshold = qualifying[label]
            if label not in qualifying and not force:
                raise RegistryError(
                    f"The live model's training report does not qualify {label}; "
                    "use --force (and optionally --threshold) to trust it anyway."
                )
            labels[label] = max(float(threshold or self.min_confidence), self.min_confidence)
        pointer["trusted"] = labels
        return self._save_pointer(pointer, by, f"{'trust' if trusted else 'untrust'} {label}")

    def rollback(self, by: str) -> dict:
        """Return to the previous live model with the labels it trusted then."""
        pointer = self.pointer()
        if not pointer or not pointer.get("previous"):
            raise RegistryError("There is no earlier model to roll back to.")
        previous, *rest = pointer["previous"]
        if not self.model_path(previous["version"]).is_file():
            raise RegistryError(f"Model file of {previous['version']} is missing.")
        pointer["version"], pointer["trusted"], pointer["previous"] = (
            previous["version"], previous["trusted"], rest,
        )
        return self._save_pointer(pointer, by, f"rollback to {previous['version']}")
