"""Shared helpers for normalizing trading symbols across profiles."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from bot_v2.orchestration.configuration import Profile

from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings

logger = logging.getLogger(__name__)

PERPS_ALLOWLIST = frozenset({"BTC-PERP", "ETH-PERP", "SOL-PERP", "XRP-PERP"})


@dataclass(frozen=True)
class SymbolNormalizationLog:
    """Captured log emitted during symbol normalization."""

    level: int
    message: str
    args: tuple[object, ...] = ()


def derivatives_enabled(profile: Profile, *, settings: RuntimeSettings | None = None) -> bool:
    """Determine whether derivatives trading should be enabled for the profile."""

    runtime_settings = settings or load_runtime_settings()

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

    if runtime_settings.coinbase_enable_derivatives_overridden:
        return runtime_settings.coinbase_enable_derivatives
    return True


def normalize_symbol_list(
    symbols: Sequence[str] | None,
    *,
    allow_derivatives: bool,
    quote: str,
    allowed_perps: Iterable[str] | None = None,
    fallback_bases: Sequence[str] | None = None,
) -> tuple[list[str], list[SymbolNormalizationLog]]:
    """Produce a normalised symbol list and captured log records."""

    logs: list[SymbolNormalizationLog] = []
    allowed_set = set(allowed_perps) if allowed_perps is not None else set(PERPS_ALLOWLIST)
    normalized: list[str] = []

    for raw in symbols or []:
        token = (raw or "").strip().upper()
        if not token:
            continue

        if allow_derivatives:
            if token.endswith("-PERP"):
                if token in allowed_set:
                    normalized.append(token)
                else:
                    logs.append(
                        SymbolNormalizationLog(
                            logging.WARNING,
                            "Filtering unsupported perpetual symbol %s. Allowed perps: %s",
                            (token, sorted(allowed_set)),
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
        fallback = ["BTC-PERP", "ETH-PERP"]
    else:
        if fallback_bases is None:
            from bot_v2.orchestration.configuration import TOP_VOLUME_BASES

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
    quote: str | None = None,
    settings: RuntimeSettings | None = None,
) -> tuple[list[str], bool]:
    """Normalize configured symbols, applying per-profile defaults and gating."""

    runtime_settings = settings or load_runtime_settings()
    quote_currency = (quote or runtime_settings.coinbase_default_quote).upper()
    allow_derivatives = derivatives_enabled(profile, settings=runtime_settings)
    normalized, logs = normalize_symbol_list(
        symbols,
        allow_derivatives=allow_derivatives,
        quote=quote_currency,
    )

    for record in logs:
        logger.log(record.level, record.message, *record.args)

    return normalized, allow_derivatives
