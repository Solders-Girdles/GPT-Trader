"""Circuit breaker utilities for runtime risk checks."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from .types import AnyLogger, LogEventFn


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

    def to_payload(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "action": self.action.value,
            "reason": self.reason,
            "value": float(self.value) if self.value is not None else None,
        }


@dataclass
class CircuitBreakerSnapshot:
    """Snapshot of the most recent trigger for a rule/symbol."""

    last_action: CircuitBreakerAction
    triggered_at: datetime


class CircuitBreakerState:
    """Tracks state of circuit breakers across time."""

    def __init__(self) -> None:
        self._rules: dict[str, CircuitBreakerRule] = {}
        self._triggers: dict[str, dict[str, CircuitBreakerSnapshot]] = {}

    def register_rule(self, rule: CircuitBreakerRule) -> None:
        self._rules[rule.name] = rule
        self._triggers.setdefault(rule.name, {})

    def record(
        self,
        rule_name: str,
        symbol: str,
        action: CircuitBreakerAction,
        triggered_at: datetime,
    ) -> None:
        self._triggers.setdefault(rule_name, {})[symbol] = CircuitBreakerSnapshot(
            last_action=action,
            triggered_at=triggered_at,
        )

    def get(self, rule_name: str, symbol: str) -> CircuitBreakerSnapshot | None:
        return self._triggers.get(rule_name, {}).get(symbol)

    def snapshot(self) -> dict[str, dict[str, CircuitBreakerSnapshot]]:
        return self._triggers


def check_volatility_circuit_breaker(
    *,
    symbol: str,
    recent_marks: Iterable[Decimal],
    config: Any | None = None,
    rule: CircuitBreakerRule | None = None,
    state: CircuitBreakerState | None = None,
    now: Callable[[], datetime],
    last_trigger: MutableMapping[str, datetime] | None = None,
    set_reduce_only: Callable[[bool, str], None] | None = None,
    log_event: LogEventFn | None = None,
    logger: AnyLogger,
) -> CircuitBreakerOutcome:
    """Evaluate rolling volatility and trigger progressive actions."""

    set_reduce_only = set_reduce_only or (lambda _enabled, _reason: None)
    log_event = log_event or (lambda *args, **kwargs: None)

    use_rule = rule is not None and state is not None

    if use_rule:
        assert rule is not None and state is not None
        if not rule.enabled:
            return CircuitBreakerOutcome(triggered=False, action=CircuitBreakerAction.NONE)
        window = int(rule.window)
    else:
        if config is None or not getattr(config, "enable_volatility_circuit_breaker", False):
            return CircuitBreakerOutcome(triggered=False, action=CircuitBreakerAction.NONE)
        window = int(getattr(config, "volatility_window_periods", 20))

    marks = list(recent_marks)
    if len(marks) < window:
        return CircuitBreakerOutcome(triggered=False, action=CircuitBreakerAction.NONE)

    returns: list[float] = []
    for a, b in zip(marks[-window:-1], marks[-window + 1 :], strict=False):
        if a and a > 0:
            try:
                returns.append(float((b - a) / a))
            except Exception:
                continue
    if len(returns) < max(10, window // 2):
        return CircuitBreakerOutcome(triggered=False, action=CircuitBreakerAction.NONE)

    stdev = statistics.stdev(returns) if len(returns) > 1 else 0.0
    rolling_vol = float(stdev * math.sqrt(252.0))
    rolling_vol_decimal = Decimal(f"{rolling_vol:.6f}")

    current_time = now()
    if use_rule:
        snapshot = state.get(rule.name, symbol)
        if snapshot and (current_time - snapshot.triggered_at) < rule.cooldown:
            return CircuitBreakerOutcome(
                triggered=False,
                action=CircuitBreakerAction.NONE,
                value=rolling_vol_decimal,
            )
        warn_th = float(rule.warning_threshold)
        reduce_only_th = float(rule.reduce_only_threshold)
        kill_th = float(rule.kill_switch_threshold)
    else:
        cooldown_min = int(getattr(config, "circuit_breaker_cooldown_minutes", 30))
        last_trigger_map = last_trigger or {}
        last_ts = last_trigger_map.get(symbol)
        snapshot = state.get("volatility_circuit_breaker", symbol) if state is not None else None
        if snapshot is not None:
            if last_trigger is not None:
                last_trigger[symbol] = snapshot.triggered_at
            last_ts = snapshot.triggered_at
        if last_ts and (current_time - last_ts) < timedelta(minutes=cooldown_min):
            return CircuitBreakerOutcome(
                triggered=False,
                action=CircuitBreakerAction.NONE,
                value=rolling_vol_decimal,
            )
        warn_th = float(getattr(config, "volatility_warning_threshold", 0.10))
        reduce_only_th = float(getattr(config, "volatility_reduce_only_threshold", 0.12))
        kill_th = float(getattr(config, "volatility_kill_switch_threshold", 0.15))

    outcome_action = CircuitBreakerAction.NONE

    if rolling_vol >= kill_th:
        if config is not None:
            config.kill_switch_enabled = True
        outcome_action = CircuitBreakerAction.KILL_SWITCH
    elif rolling_vol >= reduce_only_th:
        set_reduce_only(True, "volatility_circuit_breaker")
        outcome_action = CircuitBreakerAction.REDUCE_ONLY
    elif rolling_vol >= warn_th:
        outcome_action = CircuitBreakerAction.WARNING

    if outcome_action is not CircuitBreakerAction.NONE:
        if use_rule and state is not None:
            state.record(rule.name, symbol, outcome_action, current_time)
        elif state is not None:
            state.record("volatility_circuit_breaker", symbol, outcome_action, current_time)
        else:
            if last_trigger is not None:
                last_trigger[symbol] = current_time

        log_event(
            "volatility_circuit_breaker",
            {
                "symbol": symbol,
                "rolling_volatility": f"{rolling_vol:.6f}",
                "action": outcome_action.value,
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
            outcome_action.value,
        )
        return CircuitBreakerOutcome(
            triggered=True,
            action=outcome_action,
            reason=outcome_action.value,
            value=rolling_vol_decimal,
        )

    return CircuitBreakerOutcome(
        triggered=False,
        action=CircuitBreakerAction.NONE,
        value=rolling_vol_decimal,
    )


__all__ = [
    "CircuitBreakerAction",
    "CircuitBreakerOutcome",
    "CircuitBreakerRule",
    "CircuitBreakerSnapshot",
    "CircuitBreakerState",
    "check_volatility_circuit_breaker",
]
