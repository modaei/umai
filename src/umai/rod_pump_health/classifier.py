"""Downhole dynograph classifier used to decide when a card needs a fresh LLM review.

The models are scikit-learn random forests trained on labeled downhole cards (see
``train_classifier.py``). They only route work: a routine card whose class has not changed
since the last LLM review reuses that review instead of calling the LLM again.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FORMAT = "umai-dynograph-classifier-v1"
POINTS_PER_CARD = 100

# Labels with enough field-confirmed training cards to let a card skip the LLM. Every other
# label (gas interference, low fluid load, the synthetic failure patterns) goes to the LLM.
TRUSTED_LABELS = frozenset(
    {"healthy", "fluid_pound_mild", "fluid_pound_moderate", "fluid_pound_severe"}
)


def card_features(positions, loads, n: int = POINTS_PER_CARD) -> np.ndarray:
    """Turn one card into a fixed-length, size-independent shape vector plus a few raw stats.

    Must stay identical to the features the models were trained on.
    """
    pos = np.asarray(positions, float)
    load = np.asarray(loads, float)
    x = (pos - pos.min()) / max(np.ptp(pos), 1e-9)
    y = (load - load.min()) / max(np.ptp(load), 1e-9)

    # Close the loop and resample evenly by arc length.
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    t = np.linspace(0, s[-1], n, endpoint=False)
    xr = np.interp(t, s, x)
    yr = np.interp(t, s, y)

    # Consistent direction (clockwise) and starting point (bottom-left).
    area = 0.5 * np.sum(xr * np.roll(yr, -1) - np.roll(xr, -1) * yr)
    if area > 0:
        xr, yr = xr[::-1], yr[::-1]
    start = np.argmin(xr**2 + yr**2)
    xr, yr = np.roll(xr, -start), np.roll(yr, -start)

    raw = [abs(area), pos.max() - pos.min(), load.max(), load.min(), load.max() - load.min()]
    return np.concatenate([xr, yr, raw])


class DownholeClassifier:
    """Predict overall health and the detailed label, each with its confidence."""

    def __init__(self, models: dict, version: str):
        self.models = models
        self.version = version

    def classify(self, points) -> dict:
        positions = [point.position if hasattr(point, "position") else point["position"] for point in points]
        loads = [point.load if hasattr(point, "load") else point["load"] for point in points]
        features = card_features(positions, loads).reshape(1, -1)
        result = {"model_version": self.version}
        for target in ("health", "label"):
            model = self.models[target]
            probabilities = model.predict_proba(features)[0]
            best = int(probabilities.argmax())
            result[target] = str(model.classes_[best])
            result[target + "_confidence"] = round(float(probabilities[best]), 3)
        return result


@lru_cache(maxsize=4)
def load_classifier(path: str) -> DownholeClassifier | None:
    """Load the model file once; return None (LLM for every card) when it cannot be trusted."""
    if not path:
        return None
    file = Path(path)
    if not file.is_file():
        logger.warning("Dynograph classifier disabled: model file not found at %s", path)
        return None
    import warnings

    import joblib
    import sklearn
    from sklearn.exceptions import InconsistentVersionWarning

    try:
        # Saved forests are only reliable with the scikit-learn version that trained them;
        # scikit-learn warns on load when the versions differ.
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            saved = joblib.load(file)
    except InconsistentVersionWarning as warning:
        logger.warning(
            "Dynograph classifier disabled: trained with scikit-learn %s, running %s",
            warning.original_sklearn_version, warning.current_sklearn_version,
        )
        return None
    except Exception:
        logger.exception("Dynograph classifier disabled: model file could not be loaded")
        return None
    if not isinstance(saved, dict):
        logger.warning("Dynograph classifier disabled: unknown model format in %s", path)
        return None
    if saved.get("format") == FORMAT:
        if saved.get("sklearn_version") != sklearn.__version__:
            logger.warning(
                "Dynograph classifier disabled: trained with scikit-learn %s, running %s",
                saved.get("sklearn_version"), sklearn.__version__,
            )
            return None
        models = saved.get("models") or {}
    elif "format" not in saved and isinstance(saved.get("models"), dict):
        # Training-script (card matcher) file: {"models": {target: {"model": ..., "X": ...}}}.
        models = {
            target: entry.get("model") if isinstance(entry, dict) else None
            for target, entry in saved["models"].items()
        }
    else:
        logger.warning("Dynograph classifier disabled: unknown model format in %s", path)
        return None
    if not all(hasattr(models.get(target), "predict_proba") for target in ("health", "label")):
        logger.warning("Dynograph classifier disabled: health or label model missing")
        return None
    for target in ("health", "label"):
        # UMAI classifies one card per request; one core avoids starting a worker pool each time.
        estimator = models[target].steps[-1][1] if hasattr(models[target], "steps") else models[target]
        if "n_jobs" in estimator.get_params():
            estimator.set_params(n_jobs=1)
    version = hashlib.sha256(file.read_bytes()).hexdigest()[:12]
    return DownholeClassifier(models, version)
