"""Shared helpers for normalizing trading symbols across profiles."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from bot_v2.orchestration.configuration import Profile

logger = logging.getLogger(__name__)

PERPS_ALLOWLIST = frozenset({"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"})


def derivatives_enabled(profile: Profile) -> bool:
    """Determine whether derivatives trading should be enabled for the profile."""

    try:  # Local import to avoid circular at module load time.
        from bot_v2.orchestration.configuration import Profile as _Profile

        _ProfileClass: type[Profile] | None = _Profile
    except Exception:  # pragma: no cover - defensive fallback
        _ProfileClass = None

    if _ProfileClass is not None and isinstance(profile, _ProfileClass):
        if profile == _ProfileClass.SPOT:
            return False
    else:
        profile_value = str(getattr(profile, "value", profile or "")).lower()
        if profile_value == "spot":
            return False

    env_value = os.getenv("COINBASE_ENABLE_DERIVATIVES")
    if env_value is not None:
        return env_value == "1"
    return True


def normalize_symbols(
    profile: Profile, symbols: Sequence[str] | None, *, quote: str | None = None
) -> tuple[list[str], bool]:
    """Normalize configured symbols, applying per-profile defaults and gating."""

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

    normalized = list(dict.fromkeys(normalized))
    return normalized, allow_derivatives


def _default_symbols(derivatives: bool, quote: str) -> list[str]:
    if derivatives:
        return ["BTC-PERP", "ETH-PERP"]
    from bot_v2.orchestration.configuration import TOP_VOLUME_BASES

    return [f"{base}-{quote}" for base in TOP_VOLUME_BASES]
