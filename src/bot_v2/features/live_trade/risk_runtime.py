"""Runtime guard helpers extracted from the live risk manager."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from .guard_errors import (
    RiskGuardComputationError,
    RiskGuardTelemetryError,
)

AnyLogger = Any  # local alias for loose logger typing
LogEventFn = Callable[[str, dict[str, str], str], None]


class CircuitBreakerAction(Enum):
    """Actions that circuit breaker can trigger."""

    NONE = "none"
    WARNING = "warning"
    REDUCE_ONLY = "reduce_only"
    KILL_SWITCH = "kill_switch"


@dataclass
class CircuitBreakerRule:
    """Configuration for a circuit breaker rule."""

    name: str
    signal: str
    window: int
    warning_threshold: Decimal
    reduce_only_threshold: Decimal
    kill_switch_threshold: Decimal
    cooldown: timedelta
    enabled: bool = True


@dataclass
class CircuitBreakerOutcome:
    """Result of circuit breaker check."""

    triggered: bool
    action: CircuitBreakerAction
    reason: str | None = None
    value: Decimal | None = None


class CircuitBreakerState:
    """Tracks state of circuit breakers across time."""

    def __init__(self) -> None:
        self._rules: dict[str, CircuitBreakerRule] = {}
        self._triggers: dict[str, dict[str, tuple[CircuitBreakerAction, datetime]]] = {}

    def register_rule(self, rule: CircuitBreakerRule) -> None:
        """Register a circuit breaker rule."""
        self._rules[rule.name] = rule
        if rule.name not in self._triggers:
            self._triggers[rule.name] = {}

    def record(
        self,
        rule_name: str,
        symbol: str,
        action: CircuitBreakerAction,
        triggered_at: datetime,
    ) -> None:
        """Record a circuit breaker trigger."""
        if rule_name not in self._triggers:
            self._triggers[rule_name] = {}
        self._triggers[rule_name][symbol] = (action, triggered_at)

    def get(self, rule_name: str, symbol: str) -> tuple[CircuitBreakerAction, datetime] | None:
        """Get most recent trigger for a rule/symbol."""
        return self._triggers.get(rule_name, {}).get(symbol)

    def snapshot(self) -> dict[str, dict[str, tuple[CircuitBreakerAction, datetime]]]:
        """Get all triggers."""
        return self._triggers


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


def append_risk_metrics(
    *,
    event_store,
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
            qty = abs(Decimal(str(pos_data.get("qty", 0))))
            mark = Decimal(str(pos_data.get("mark", 0)))
        except Exception:
            continue
        if qty > 0 and mark > 0:
            notional = qty * mark
            total_notional += notional
            leverage = notional / equity if equity > 0 else Decimal("0")
            max_leverage = max(max_leverage, leverage)

    exposure_pct = total_notional / equity if equity > 0 else Decimal("0")
    if start_of_day_equity > 0:
        daily_pnl_pct = daily_pnl / start_of_day_equity
    else:
        daily_pnl_pct = Decimal("0")

    logger.debug(
        "Risk snapshot: equity=%s notional=%s exposure=%.3f max_lev=%.2f daily_pnl=%s daily_pnl_pct=%.4f reduce_only=%s kill=%s",
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
        event_store.append_metric(
            bot_id="risk_engine",
            metrics={
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
        )
    except Exception as exc:
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
        for sym, p in positions.items():
            qty = abs(Decimal(str(p.get("qty", 0))))
            mark = Decimal(str(p.get("mark", 0)))
            notional_vals.append(qty * mark)
        total = sum(notional_vals) if notional_vals else Decimal("0")
        if total <= 0:
            return False
        hhi = sum((v / total) ** 2 for v in notional_vals)
        if hhi > Decimal("0.4"):
            log_event("concentration_risk", {"hhi": str(hhi)}, guard="correlation_risk")
            logger.warning("Concentration risk detected (HHI=%.3f)", hhi)
            return True

        # Correlation placeholder – left for future enrichment
        return False
    except Exception as exc:
        raise RiskGuardComputationError(
            guard="correlation_risk",
            message="Failed to evaluate correlation risk",
            details={"symbols": list(positions.keys())},
            original=exc,
        ) from exc


def check_volatility_circuit_breaker(
    *,
    symbol: str,
    recent_marks: Iterable[Decimal],
    config: Any,
    now: Callable[[], datetime],
    last_trigger: MutableMapping[str, datetime],
    set_reduce_only: Callable[[bool, str], None],
    log_event: LogEventFn,
    logger: AnyLogger,
) -> dict[str, Any]:
    """Evaluate rolling volatility and trigger progressive actions."""
    if not getattr(config, "enable_volatility_circuit_breaker", False):
        return {"triggered": False}

    window = int(getattr(config, "volatility_window_periods", 20))
    marks = list(recent_marks)
    if len(marks) < window:
        return {"triggered": False}

    rets: list[float] = []
    for a, b in zip(marks[-window:-1], marks[-window + 1 :], strict=False):
        if a and a > 0:
            try:
                rets.append(float((b - a) / a))
            except Exception:
                continue
    if len(rets) < max(10, window // 2):
        return {"triggered": False}

    stdev = statistics.stdev(rets) if len(rets) > 1 else 0.0
    rolling_vol = float(stdev * math.sqrt(252.0))

    cooldown_min = int(getattr(config, "circuit_breaker_cooldown_minutes", 30))
    last_ts = last_trigger.get(symbol)
    current_time = now()
    if last_ts and (current_time - last_ts) < timedelta(minutes=cooldown_min):
        return {"triggered": False, "volatility": rolling_vol}

    warn_th = float(getattr(config, "volatility_warning_threshold", 0.10))
    reduce_only_th = float(getattr(config, "volatility_reduce_only_threshold", 0.12))
    kill_th = float(getattr(config, "volatility_kill_switch_threshold", 0.15))

    action: str | None = None
    if rolling_vol >= kill_th:
        config.kill_switch_enabled = True
        action = "kill_switch"
    elif rolling_vol >= reduce_only_th:
        set_reduce_only(True, "volatility_circuit_breaker")
        action = "reduce_only"
    elif rolling_vol >= warn_th:
        action = "warning"

    if action:
        last_trigger[symbol] = current_time
        log_event(
            "volatility_circuit_breaker",
            {
                "symbol": symbol,
                "rolling_volatility": f"{rolling_vol:.6f}",
                "action": action,
                "warning_threshold": warn_th,
                "reduce_only_threshold": reduce_only_th,
                "kill_switch_threshold": kill_th,
            },
            guard="volatility_circuit_breaker",
        )
        logger.warning(
            "Volatility CB: %s vol=%.3f action=%s",
            symbol,
            rolling_vol,
            action,
        )
        return {"triggered": True, "action": action, "volatility": rolling_vol}

    return {"triggered": False, "volatility": rolling_vol}
