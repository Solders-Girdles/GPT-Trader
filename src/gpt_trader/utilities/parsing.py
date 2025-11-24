"""Common parsing helpers for normalising primitive configuration values."""

from __future__ import annotations

from typing import Final

TRUE_BOOLEAN_TOKENS: Final[frozenset[str]] = frozenset({"1", "true", "t", "yes", "y", "on"})
FALSE_BOOLEAN_TOKENS: Final[frozenset[str]] = frozenset({"0", "false", "f", "no", "n", "off"})


def interpret_tristate_bool(value: str | None) -> bool | None:
    """Return ``True``/``False`` for recognised boolean tokens otherwise ``None``.

    The helper treats empty or whitespace-only strings the same as ``None`` so callers
    can pass-through unset environment variables without additional sanitisation.
    """

    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    lowered = normalized.lower()
    if lowered in TRUE_BOOLEAN_TOKENS:
        return True
    if lowered in FALSE_BOOLEAN_TOKENS:
        return False
    return None


__all__ = [
    "FALSE_BOOLEAN_TOKENS",
    "TRUE_BOOLEAN_TOKENS",
    "interpret_tristate_bool",
]
