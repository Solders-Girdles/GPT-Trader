"""Shared risk calculations and schedule helpers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from datetime import time as dtime
from decimal import Decimal
from typing import Any

RiskInfoProvider = Callable[[str], dict[str, Any]]


def evaluate_daytime_window(
    config: Any,
    now: datetime | None,
    *,
    logger: Any | None = None,
) -> bool | None:
    """Return True if ``now`` is within the configured daytime window."""
    start_s = getattr(config, "daytime_start_utc", None)
    end_s = getattr(config, "daytime_end_utc", None)
    if not start_s or not end_s:
        return None
    try:
        start_h, start_m = map(int, start_s.split(":"))
        end_h, end_m = map(int, end_s.split(":"))
        start = dtime(start_h, start_m)
        end = dtime(end_h, end_m)
        current = now or datetime.utcnow()
        t = current.time()
        if start <= end:
            return start <= t < end
        # window spans midnight
        return t >= start or t < end
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            logger.debug("Failed to evaluate daytime window: %s", exc, exc_info=True)
        return None


def effective_symbol_leverage_cap(
    symbol: str,
    config: Any,
    *,
    now: datetime | None,
    risk_info_provider: RiskInfoProvider | None,
    logger: Any | None = None,
) -> int:
    """Compute the effective leverage cap considering schedules and provider overrides."""
    cap = config.leverage_max_per_symbol.get(symbol, config.max_leverage)

    is_day = evaluate_daytime_window(config, now, logger=logger)
    try:
        day_caps = getattr(config, "day_leverage_max_per_symbol", {})
        night_caps = getattr(config, "night_leverage_max_per_symbol", {})
        if is_day is True and symbol in day_caps:
            cap = min(cap, int(day_caps[symbol]))
        elif is_day is False and symbol in night_caps:
            cap = min(cap, int(night_caps[symbol]))
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            logger.debug(
                "Failed to apply day/night leverage override for %s: %s", symbol, exc, exc_info=True
            )

    if risk_info_provider is not None:
        try:
            info = risk_info_provider(symbol) or {}
            provider_cap = info.get("max_leverage") or info.get("leverage_cap")
            if provider_cap is not None:
                cap = min(cap, int(provider_cap))
        except Exception as exc:  # pragma: no cover - defensive logging
            if logger is not None:
                logger.debug(
                    "Risk info provider failed for leverage cap on %s: %s",
                    symbol,
                    exc,
                    exc_info=True,
                )
    return int(cap)


def effective_mmr(
    symbol: str,
    config: Any,
    *,
    now: datetime | None,
    risk_info_provider: RiskInfoProvider | None,
    logger: Any | None = None,
) -> Decimal:
    """Compute the maintenance margin rate with provider/schedule overrides."""
    if risk_info_provider is not None:
        try:
            info = risk_info_provider(symbol) or {}
            raw = info.get("maintenance_margin_rate") or info.get("mmr")
            if raw is not None:
                return Decimal(str(raw))
        except Exception as exc:  # pragma: no cover - defensive logging
            if logger is not None:
                logger.debug(
                    "Risk info provider failed for MMR on %s: %s", symbol, exc, exc_info=True
                )

    is_day = evaluate_daytime_window(config, now, logger=logger)
    try:
        day_mmr = getattr(config, "day_mmr_per_symbol", {})
        night_mmr = getattr(config, "night_mmr_per_symbol", {})
        if is_day is True and symbol in day_mmr:
            return Decimal(str(day_mmr[symbol]))
        if is_day is False and symbol in night_mmr:
            return Decimal(str(night_mmr[symbol]))
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger is not None:
            logger.debug(
                "Failed to apply day/night MMR override for %s: %s", symbol, exc, exc_info=True
            )

    return Decimal(str(config.default_maintenance_margin_rate))
