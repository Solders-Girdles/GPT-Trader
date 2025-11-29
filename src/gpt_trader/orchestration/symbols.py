"""Shared helpers for normalizing trading symbols across profiles."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from gpt_trader.orchestration.configuration import BotConfig, Profile

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="symbols")

PERPS_ALLOWLIST = frozenset({"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"})
US_FUTURES_ALLOWLIST = frozenset({"BTC-FUTURES", "ETH-FUTURES", "SOL-FUTURES", "XRP-FUTURES"})


@dataclass(frozen=True)
class SymbolNormalizationLog:
    """Captured log emitted during symbol normalization."""

    level: int
    message: str
    args: tuple[object, ...] = ()


def derivatives_enabled(profile: Profile, *, config: BotConfig) -> bool:
    """Determine whether derivatives trading should be enabled for the profile."""

    try:  # Local import to avoid circular at module load time.
        from gpt_trader.orchestration.configuration import Profile as _Profile

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

    # Use strict config toggle
    return config.derivatives_enabled


def us_futures_enabled(profile: Profile, *, config: BotConfig) -> bool:
    """Determine whether US futures trading should be enabled for the profile."""

    # Check if derivatives are enabled at all
    if not derivatives_enabled(profile, config=config):
        return False

    # Check US futures specific flag
    if config.coinbase_us_futures_enabled:
        return True

    # Check derivatives type
    if config.coinbase_derivatives_type == "us_futures":
        return True

    return False


def intx_perpetuals_enabled(profile: Profile, *, config: BotConfig) -> bool:
    """Determine whether INTX perpetuals trading should be enabled for the profile."""

    # Check if derivatives are enabled at all
    if not derivatives_enabled(profile, config=config):
        return False

    # Check INTX perpetuals specific flag
    if config.coinbase_intx_perpetuals_enabled:
        return True

    # Check derivatives type (default to INTX)
    if config.coinbase_derivatives_type in ("intx_perps", "perpetuals"):
        return True

    return True


def normalize_symbol_list(
    symbols: Sequence[str] | None,
    *,
    allow_derivatives: bool,
    quote: str,
    allowed_perps: Iterable[str] | None = None,
    allowed_us_futures: Iterable[str] | None = None,
    fallback_bases: Sequence[str] | None = None,
) -> tuple[list[str], list[SymbolNormalizationLog]]:
    """Produce a normalised symbol list and captured log records."""

    logs: list[SymbolNormalizationLog] = []
    allowed_perps_set = set(allowed_perps) if allowed_perps is not None else set(PERPS_ALLOWLIST)
    allowed_us_futures_set = (
        set(allowed_us_futures) if allowed_us_futures is not None else set(US_FUTURES_ALLOWLIST)
    )
    normalized: list[str] = []

    for raw in symbols or []:
        token = (raw or "").strip().upper()
        if not token:
            continue

        if allow_derivatives:
            if token.endswith("-PERP"):
                if token in allowed_perps_set:
                    normalized.append(token)
                else:
                    logs.append(
                        SymbolNormalizationLog(
                            logging.WARNING,
                            "Filtering unsupported perpetual symbol %s. Allowed perps: %s",
                            (token, sorted(allowed_perps_set)),
                        )
                    )
            elif token.endswith("-FUTURES"):
                if token in allowed_us_futures_set:
                    normalized.append(token)
                else:
                    logs.append(
                        SymbolNormalizationLog(
                            logging.WARNING,
                            "Filtering unsupported US futures symbol %s. Allowed US futures: %s",
                            (token, sorted(allowed_us_futures_set)),
                        )
                    )
            else:
                normalized.append(token)
            continue

        if token.endswith("-PERP"):
            base = token.split("-", 1)[0]
            replacement = f"{base}-{quote}"
            logs.append(
                SymbolNormalizationLog(
                    logging.WARNING,
                    "Derivatives disabled. Replacing %s with spot symbol %s",
                    (token, replacement),
                )
            )
            token = replacement

        normalized.append(token)

    normalized = list(dict.fromkeys(normalized))
    if normalized:
        return normalized, logs

    if allow_derivatives:
        fallback = ["BTC-PERP", "ETH-PERP", "BTC-FUTURES", "ETH-FUTURES"]
    else:
        if fallback_bases is None:
            from gpt_trader.orchestration.configuration import TOP_VOLUME_BASES

            fallback_bases = TOP_VOLUME_BASES
        fallback = [f"{base}-{quote}" for base in fallback_bases]

    logs.append(
        SymbolNormalizationLog(
            logging.INFO,
            "No valid symbols provided. Falling back to %s",
            (fallback,),
        )
    )
    return fallback, logs


def normalize_symbols(
    profile: Profile,
    symbols: Sequence[str] | None,
    *,
    config: BotConfig,
    quote: str | None = None,
) -> tuple[list[str], bool]:
    """Normalize configured symbols, applying per-profile defaults and gating."""

    quote_currency = (quote or config.coinbase_default_quote).upper()
    allow_derivatives = derivatives_enabled(profile, config=config)
    normalized, logs = normalize_symbol_list(
        symbols,
        allow_derivatives=allow_derivatives,
        quote=quote_currency,
    )

    for record in logs:
        logger.log(record.level, record.message, *record.args)

    return normalized, allow_derivatives
