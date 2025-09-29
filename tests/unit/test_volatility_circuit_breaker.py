import math
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.live_trade.risk_runtime import CircuitBreakerAction


def generate_marks_with_volatility(target_annual_vol: float, n: int = 30) -> list[Decimal]:
    """Generate synthetic price series with target annualized volatility.

    Uses alternating returns +/- s where s = target_vol / sqrt(252).
    """
    if n < 25:
        n = 25
    s = target_annual_vol / math.sqrt(252.0)
    price = 100.0
    marks = [Decimal(str(price))]
    for i in range(n - 1):
        r = s if i % 2 == 0 else -s
        price = max(0.01, price * (1.0 + r))
        marks.append(Decimal(str(price)))
    return marks


def make_manager_with_thresholds(warn: float, reduce_only: float, kill: float) -> LiveRiskManager:
    config = RiskConfig()
    config.enable_volatility_circuit_breaker = True
    config.volatility_warning_threshold = warn
    config.volatility_reduce_only_threshold = reduce_only
    config.volatility_kill_switch_threshold = kill
    config.volatility_window_periods = 20
    return LiveRiskManager(config=config)


def test_progressive_thresholds():
    rm = make_manager_with_thresholds(0.10, 0.12, 0.15)

    sym = "BTC-PERP"

    # Warning
    marks_warn = generate_marks_with_volatility(0.10, n=30)
    res = rm.check_volatility_circuit_breaker(sym, marks_warn)
    assert res.triggered is True
    assert res.action is CircuitBreakerAction.WARNING
    assert rm.config.reduce_only_mode is False
    assert rm.config.kill_switch_enabled is False

    # Reduce-only
    marks_red = generate_marks_with_volatility(0.12, n=30)
    rm.circuit_breaker_state.record(
        "volatility_circuit_breaker",
        sym,
        CircuitBreakerAction.WARNING,
        triggered_at=datetime.utcnow()
        - timedelta(minutes=rm.config.circuit_breaker_cooldown_minutes + 1),
    )
    res = rm.check_volatility_circuit_breaker(sym, marks_red)
    assert res.triggered is True
    assert res.action is CircuitBreakerAction.REDUCE_ONLY
    assert rm.config.reduce_only_mode is True
    assert rm.config.kill_switch_enabled is False

    # Kill switch
    marks_kill = generate_marks_with_volatility(0.15, n=30)
    rm.circuit_breaker_state.record(
        "volatility_circuit_breaker",
        sym,
        CircuitBreakerAction.WARNING,
        triggered_at=datetime.utcnow()
        - timedelta(minutes=rm.config.circuit_breaker_cooldown_minutes + 1),
    )
    res = rm.check_volatility_circuit_breaker(sym, marks_kill)
    assert res.triggered is True
    assert res.action is CircuitBreakerAction.KILL_SWITCH
    assert rm.config.kill_switch_enabled is True


def test_cooldown_behavior():
    config = RiskConfig()
    config.enable_volatility_circuit_breaker = True
    config.volatility_warning_threshold = 0.05
    config.volatility_reduce_only_threshold = 0.06
    config.volatility_kill_switch_threshold = 0.07
    config.volatility_window_periods = 20
    config.circuit_breaker_cooldown_minutes = 5
    rm = LiveRiskManager(config=config)

    marks = generate_marks_with_volatility(0.16, n=30)

    # First trigger
    res1 = rm.check_volatility_circuit_breaker("BTC-PERP", marks)
    assert res1.triggered is True

    # Second trigger within cooldown should be suppressed
    res2 = rm.check_volatility_circuit_breaker("BTC-PERP", marks)
    assert res2.triggered is False

    # Advance cooldown by manually backdating last trigger
    rm.circuit_breaker_state.record(
        "volatility_circuit_breaker",
        "BTC-PERP",
        CircuitBreakerAction.WARNING,
        triggered_at=datetime.utcnow()
        - timedelta(minutes=config.circuit_breaker_cooldown_minutes + 1),
    )
    res3 = rm.check_volatility_circuit_breaker("BTC-PERP", marks)
    assert res3.triggered is True
