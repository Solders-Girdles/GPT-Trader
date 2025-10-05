"""Pure symbol utilities for normalization and validation.

This module contains shared helpers for symbol processing that don't
depend on orchestration layer modules, breaking circular dependencies.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Allowlist of supported perpetual trading symbols
PERPS_ALLOWLIST = frozenset({"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"})

# Top volume base currencies for spot trading defaults
TOP_VOLUME_BASES = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "LTC",
    "ADA",
    "DOGE",
    "BCH",
    "AVAX",
    "LINK",
]


def derivatives_enabled(profile: Any) -> bool:
    """Determine whether derivatives trading should be enabled for the profile.

    Args:
        profile: Trading profile (duck-typed to avoid circular imports)

    Returns:
        True if derivatives are enabled, False otherwise
    """
    # Extract profile value as string for comparison
    profile_value = str(getattr(profile, "value", profile or "")).lower()

    # Disable derivatives for SPOT profile
    if profile_value == "spot":
        return False

    # Check environment variable
    return os.getenv("COINBASE_ENABLE_DERIVATIVES", "0") == "1"


def normalize_symbols(
    profile: Any, symbols: Sequence[str] | None, *, quote: str | None = None
) -> tuple[list[str], bool]:
    """Normalize configured symbols, applying per-profile defaults and gating.

    Args:
        profile: Trading profile (duck-typed to avoid circular imports)
        symbols: List of symbols to normalize
        quote: Quote currency (defaults to COINBASE_DEFAULT_QUOTE or USD)

    Returns:
        Tuple of (normalized symbols list, derivatives_enabled flag)
    """
    quote_currency = (quote or os.getenv("COINBASE_DEFAULT_QUOTE") or "USD").upper()
    allow_derivatives = derivatives_enabled(profile)
    normalized: list[str] = []

    for raw in symbols or []:
        token = (raw or "").strip().upper()
        if not token:
            continue

        if token.endswith("-PERP"):
            if allow_derivatives:
                if token in PERPS_ALLOWLIST:
                    normalized.append(token)
                else:
                    logger.warning(
                        "Filtering unsupported perpetual symbol %s. Allowed perps: %s",
                        token,
                        sorted(PERPS_ALLOWLIST),
                    )
            else:
                base = token.split("-", 1)[0]
                replacement = f"{base}-{quote_currency}"
                logger.warning(
                    "Derivatives disabled. Replacing %s with spot symbol %s",
                    token,
                    replacement,
                )
                normalized.append(replacement)
        else:
            normalized.append(token)

    if not normalized:
        normalized = _default_symbols(allow_derivatives, quote_currency)
        logger.info("No valid symbols provided. Falling back to %s", normalized)

    # Remove duplicates while preserving order
    normalized = list(dict.fromkeys(normalized))
    return normalized, allow_derivatives


def _default_symbols(derivatives: bool, quote: str) -> list[str]:
    """Generate default symbols based on profile and quote currency.

    Args:
        derivatives: Whether derivatives are enabled
        quote: Quote currency

    Returns:
        List of default trading symbols
    """
    if derivatives:
        return ["BTC-PERP", "ETH-PERP"]
    return [f"{base}-{quote}" for base in TOP_VOLUME_BASES]
