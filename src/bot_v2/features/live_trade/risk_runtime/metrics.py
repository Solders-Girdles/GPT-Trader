"""Telemetry helpers for runtime risk monitoring."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.guard_errors import RiskGuardTelemetryError
from bot_v2.utilities.telemetry import emit_metric

from .types import AnyLogger


def append_risk_metrics(
    *,
    event_store: Any,
    now: Callable[[], datetime],
    equity: Decimal,
    positions: dict[str, Any],
    daily_pnl: Decimal,
    start_of_day_equity: Decimal,
    reduce_only: bool,
    kill_switch_enabled: bool,
    logger: AnyLogger,
) -> None:
    """Persist a summary of current risk posture."""

    total_notional = Decimal("0")
    max_leverage = Decimal("0")

    for symbol, pos_data in positions.items():
        try:
            qty = abs(Decimal(str(pos_data.get("quantity", pos_data.get("qty", 0)))))
            mark = Decimal(str(pos_data.get("mark", 0)))
        except Exception:
            continue
        if qty > 0 and mark > 0:
            notional = qty * mark
            total_notional += notional
            leverage = notional / equity if equity > 0 else Decimal("0")
            max_leverage = max(max_leverage, leverage)

    exposure_pct = total_notional / equity if equity > 0 else Decimal("0")
    daily_pnl_pct = daily_pnl / start_of_day_equity if start_of_day_equity > 0 else Decimal("0")

    logger.debug(
        "Risk snapshot: equity=%s notional=%s exposure=%.3f max_lev=%.2f daily_pnl=%s "
        "daily_pnl_pct=%.4f reduce_only=%s kill=%s",
        equity,
        total_notional,
        exposure_pct,
        max_leverage,
        daily_pnl,
        daily_pnl_pct,
        reduce_only,
        kill_switch_enabled,
    )

    try:
        emit_metric(
            event_store,
            "risk_engine",
            {
                "event_type": "risk_snapshot",
                "timestamp": now().isoformat(),
                "equity": str(equity),
                "total_notional": str(total_notional),
                "exposure_pct": str(exposure_pct),
                "max_leverage": str(max_leverage),
                "daily_pnl": str(daily_pnl),
                "daily_pnl_pct": str(daily_pnl_pct),
                "reduce_only": str(reduce_only),
                "kill_switch": str(kill_switch_enabled),
            },
            raise_on_error=True,
            logger=logger,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RiskGuardTelemetryError(
            guard="risk_metrics",
            message="Failed to persist risk snapshot metric",
            details={
                "equity": str(equity),
                "total_notional": str(total_notional),
                "exposure_pct": str(exposure_pct),
            },
            original=exc,
        ) from exc


__all__ = ["append_risk_metrics"]
