import math
from decimal import Decimal

import pytest

from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.config.live_trade_config import RiskConfig


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
    cfg = RiskConfig()
    cfg.enable_volatility_circuit_breaker = True
    cfg.volatility_warning_threshold = warn
    cfg.volatility_reduce_only_threshold = reduce_only
    cfg.volatility_kill_switch_threshold = kill
    cfg.volatility_window_periods = 20
    return LiveRiskManager(config=cfg)


def test_progressive_thresholds():
    rm = make_manager_with_thresholds(0.10, 0.12, 0.15)

    sym = "BTC-PERP"

    # Warning
    marks_warn = generate_marks_with_volatility(0.10, n=30)
    res = rm.check_volatility_circuit_breaker(sym, marks_warn)
    assert res.get("triggered") is True
    assert res.get("action") == "warning"
    assert rm.config.reduce_only_mode is False
    assert rm.config.kill_switch_enabled is False

    # Reduce-only
    marks_red = generate_marks_with_volatility(0.12, n=30)
    # backdate cooldown
    from datetime import datetime, timedelta
    rm._cb_last_trigger[sym] = datetime.utcnow() - timedelta(minutes=rm.config.circuit_breaker_cooldown_minutes + 1)
    res = rm.check_volatility_circuit_breaker(sym, marks_red)
    assert res.get("triggered") is True
    assert res.get("action") == "reduce_only"
    assert rm.config.reduce_only_mode is True
    assert rm.config.kill_switch_enabled is False

    # Kill switch
    marks_kill = generate_marks_with_volatility(0.15, n=30)
    rm._cb_last_trigger[sym] = datetime.utcnow() - timedelta(minutes=rm.config.circuit_breaker_cooldown_minutes + 1)
    res = rm.check_volatility_circuit_breaker(sym, marks_kill)
    assert res.get("triggered") is True
    assert res.get("action") == "kill_switch"
    assert rm.config.kill_switch_enabled is True


def test_cooldown_behavior():
    cfg = RiskConfig()
    cfg.enable_volatility_circuit_breaker = True
    cfg.volatility_warning_threshold = 0.05
    cfg.volatility_reduce_only_threshold = 0.06
    cfg.volatility_kill_switch_threshold = 0.07
    cfg.volatility_window_periods = 20
    cfg.circuit_breaker_cooldown_minutes = 5
    rm = LiveRiskManager(config=cfg)

    marks = generate_marks_with_volatility(0.16, n=30)

    # First trigger
    res1 = rm.check_volatility_circuit_breaker("BTC-PERP", marks)
    assert res1.get("triggered") is True

    # Second trigger within cooldown should be suppressed
    res2 = rm.check_volatility_circuit_breaker("BTC-PERP", marks)
    assert res2.get("triggered") is False

    # Advance cooldown by manually backdating last trigger
    from datetime import datetime, timedelta

    rm._cb_last_trigger["BTC-PERP"] = datetime.utcnow() - timedelta(minutes=cfg.circuit_breaker_cooldown_minutes + 1)
    res3 = rm.check_volatility_circuit_breaker("BTC-PERP", marks)
    assert res3.get("triggered") is True
