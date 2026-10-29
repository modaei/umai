"""Downhole card patterns shared by the LLM review, the classifier and its training.

The LLM names one of these patterns for every card it reviews; UMetriqs stores that pattern as a
training label, so the classifier and the LLM speak the same vocabulary.
"""

from __future__ import annotations

# Pattern -> (health class used by the classifier, plain-words meaning given to the LLM).
PATTERNS = {
    "healthy": ("Healthy", "full or near-full pump card, normal valve action"),
    "fluid_pound_mild": ("Watch", "fluid pound, pump fillage 75-90%"),
    "fluid_pound_moderate": (
        "Problem", "fluid pound, pump fillage 60 to under 75%; plunger hits fluid on the downstroke",
    ),
    "fluid_pound_severe": ("Problem", "fluid pound, pump fillage under 60%; hard impact, pumped off"),
    "gas_interference_moderate": (
        "Problem", "gas compressing in the barrel delays valve opening, rounded unloading; fillage 60% or more",
    ),
    "gas_interference_severe": (
        "Problem", "a large part of the downstroke is spent compressing gas; fillage under 60%",
    ),
    "gas_lock": ("Problem", "pump gas-locked; almost no fluid load is picked up or released"),
    "slow_upstroke_loading": ("Watch", "pump takes long to pick up the fluid load on the upstroke"),
    "low_fluid_load": (
        "Problem", "card collapsed, pump not lifting a full column (rod part, unseated or worn pump, tubing leak)",
    ),
    "traveling_valve_leak": ("Problem", "load falls off during the upstroke"),
    "standing_valve_leak": ("Problem", "load rises during the downstroke"),
    "pump_tagging_down": ("Problem", "load spike at the bottom of the downstroke"),
    "pump_tagging_up": ("Problem", "load spike at the top of the upstroke"),
    "rod_part": ("Problem", "card collapsed to the rod weight, no fluid load"),
    "unseated_pump": ("Problem", "narrow card near the rod weight; pump not seated"),
    "sticking_plunger": ("Problem", "irregular load spikes where the plunger sticks"),
    "bent_barrel": ("Problem", "load bump at the same position on each stroke"),
    "split_barrel": ("Problem", "fluid load lost over part of the upstroke"),
    "unanchored_tubing": ("Problem", "sloped load pick-up and release from tubing movement"),
    "fluid_friction_drag": ("Problem", "wide card from fluid friction or paraffin drag"),
    "abnormal_shape": ("Bad card", "erratic card with no clear valve action; transient or bad data"),
    "other": ("Problem", "a clear pattern that is none of the above"),
}

# Older label names kept in imported training data, mapped to their current pattern.
ALIASES = {"tv_leak_suspect": "traveling_valve_leak"}

# Patterns that never train the classifier: they do not name a learnable card shape.
UNTRAINED_PATTERNS = frozenset({"other"})

# Labels trusted before trust was kept with each model in the classifier registry.
LEGACY_TRUSTED_LABELS = frozenset(
    {"healthy", "fluid_pound_mild", "fluid_pound_moderate", "fluid_pound_severe"}
)


def pattern_guide() -> str:
    """The pattern list as prompt text: one line per pattern with its meaning."""
    return "\n".join(f"- {name}: {meaning}" for name, (_, meaning) in PATTERNS.items())
