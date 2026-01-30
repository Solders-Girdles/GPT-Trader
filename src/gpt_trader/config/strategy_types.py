"""Canonical strategy type names and alias resolution."""

from __future__ import annotations

CANONICAL_STRATEGY_TYPES: tuple[str, ...] = (
    "baseline",
    "mean_reversion",
    "ensemble",
    "regime_switcher",
)

STRATEGY_TYPE_ALIASES: dict[str, str] = {
    "perps_baseline": "baseline",
    "spot": "baseline",
}

STRATEGY_VARIANTS: dict[str, str] = {
    "perps_baseline": "perps",
    "spot": "spot",
}


def normalize_strategy_type(value: str) -> str:
    """Return canonical strategy type for a user-provided value."""
    return STRATEGY_TYPE_ALIASES.get(value, value)


def resolve_strategy_type(value: str, *, variant: str | None = None) -> tuple[str, str | None]:
    """Resolve canonical strategy type and variant for CLI/config inputs."""
    canonical = normalize_strategy_type(value)
    resolved_variant = variant or STRATEGY_VARIANTS.get(value)
    if canonical == "baseline" and resolved_variant is None:
        resolved_variant = "perps"
    return canonical, resolved_variant


def is_known_strategy_type(value: str) -> bool:
    """Check whether a value is a known canonical strategy type or alias."""
    return normalize_strategy_type(value) in CANONICAL_STRATEGY_TYPES

