"""Risk guard helpers for live trading."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.guard_errors import (
    RiskGuardComputationError,
)

from .types import AnyLogger, LogEventFn


def check_mark_staleness(
    *,
    symbol: str,
    last_mark_update: MutableMapping[str, datetime],
    now: Callable[[], datetime],
    max_staleness_seconds: int,
    log_event: LogEventFn,
    logger: AnyLogger,
) -> bool:
    """Return True when mark data is stale enough to halt trading."""

    if symbol not in last_mark_update:
        return False

    age = now() - last_mark_update[symbol]
    soft_limit = timedelta(seconds=max_staleness_seconds)
    hard_limit = timedelta(seconds=max_staleness_seconds * 2)

    if age > hard_limit:
        log_event(
            "stale_mark_price",
            {
                "symbol": symbol,
                "age_seconds": str(age.total_seconds()),
                "limit_seconds": str(max_staleness_seconds),
                "action": "halt_new_orders",
            },
            guard="mark_staleness",
        )
        logger.warning(
            "Stale mark price for %s: %.0fs > hard limit %.0fs - Halting new orders",
            symbol,
            age.total_seconds(),
            hard_limit.total_seconds(),
        )
        return True
    if age > soft_limit:
        logger.info(
            "Mark slightly stale for %s: %.0fs > %.0fs - continuing",
            symbol,
            age.total_seconds(),
            soft_limit.total_seconds(),
        )
    return False


def check_correlation_risk(
    positions: dict[str, Any],
    *,
    log_event: LogEventFn,
    logger: AnyLogger,
) -> bool:
    """Return True when portfolio concentration or correlation limits are breached."""

    try:
        symbols = list(positions.keys())
        if len(symbols) < 2:
            return False

        notional_vals: list[Decimal] = []
        for sym, payload in positions.items():
            qty = abs(Decimal(str(payload.get("quantity", payload.get("qty", 0)))))
            mark = Decimal(str(payload.get("mark", 0)))
            notional_vals.append(qty * mark)
        total = sum(notional_vals) if notional_vals else Decimal("0")
        if total <= 0:
            return False
        hhi = sum((value / total) ** 2 for value in notional_vals)
        if hhi > Decimal("0.4"):
            log_event("concentration_risk", {"hhi": str(hhi)}, guard="correlation_risk")
            logger.warning("Concentration risk detected (HHI=%.3f)", hhi)
            return True

        # Correlation placeholder – left for future enrichment
        return False
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RiskGuardComputationError(
            guard="correlation_risk",
            message="Failed to evaluate correlation risk",
            details={"symbols": list(positions.keys())},
            original=exc,
        ) from exc


__all__ = ["check_mark_staleness", "check_correlation_risk"]
