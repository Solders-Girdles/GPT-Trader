"""Runtime guard tests for LiveRiskManager and shared helpers."""

import datetime as dt
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.features.live_trade.risk_runtime import (
    CircuitBreakerAction,
    CircuitBreakerRule,
    CircuitBreakerState,
    append_risk_metrics,
    check_correlation_risk,
    check_mark_staleness as runtime_check_mark_staleness,
    check_volatility_circuit_breaker as runtime_check_volatility,
)


class DummyLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []
        self.debugs = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        self.infos.append(msg % args if args else msg)

    def debug(self, msg, *args):
        self.debugs.append(msg % args if args else msg)


class DummyEventStore:
    def __init__(self):
        self.metrics = []

    def append_metric(self, **kwargs):
        self.metrics.append(kwargs)


class TestMarkStaleness:
    """Short-term mark price staleness checks."""

    @pytest.fixture
    def risk_manager(self):
        config = RiskConfig(max_mark_staleness_seconds=180)
        return LiveRiskManager(config=config)

    def test_fresh_mark_detection(self, risk_manager):
        is_stale = risk_manager.check_mark_staleness(
            symbol="BTC-PERP",
            mark_timestamp=datetime.utcnow(),
        )
        assert is_stale is False

    def test_stale_mark_detection(self, risk_manager):
        is_stale = risk_manager.check_mark_staleness(
            symbol="BTC-PERP",
            mark_timestamp=datetime.utcnow() - timedelta(minutes=7),
        )
        assert is_stale is True

    def test_helper_triggers_event(self):
        last_update = {"BTC-USD": dt.datetime.utcnow() - dt.timedelta(seconds=5)}
        events = []

        def log_event(event_type, details, *, guard):
            events.append((event_type, details, guard))

        logger = DummyLogger()
        result = runtime_check_mark_staleness(
            symbol="BTC-USD",
            last_mark_update=last_update,
            now=lambda: dt.datetime.utcnow(),
            max_staleness_seconds=1,
            log_event=log_event,
            logger=logger,
        )
        assert result is True
        assert any(evt[0] == "stale_mark_price" for evt in events)
        assert logger.warnings


class TestVolatilityCircuitBreaker:
    """Volatility guard behavior across thresholds."""

    @pytest.fixture
    def risk_manager(self):
        config = RiskConfig(
            enable_volatility_circuit_breaker=True,
            volatility_warning_threshold=0.15,
            volatility_reduce_only_threshold=0.20,
            volatility_kill_switch_threshold=0.25,
        )
        return LiveRiskManager(config=config)

    def test_low_volatility_allows_trading(self, risk_manager):
        base = Decimal("50000")
        marks = [base + Decimal(i) for i in range(25)]
        result = risk_manager.check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=marks,
        )
        assert result.triggered is False
        assert result.action is CircuitBreakerAction.NONE

    def test_high_volatility_triggers_warning(self, risk_manager):
        base = Decimal("50000")
        swings = [Decimal("8000"), Decimal("-8000")] * 13
        marks = [base]
        for delta in swings:
            marks.append(marks[-1] + delta)

        result = risk_manager.check_volatility_circuit_breaker(
            symbol="BTC-PERP",
            recent_marks=marks,
        )
        assert result.triggered is True
        assert result.action in {
            CircuitBreakerAction.WARNING,
            CircuitBreakerAction.REDUCE_ONLY,
            CircuitBreakerAction.KILL_SWITCH,
        }

    def test_helper_reduce_only_flow(self):
        rule = CircuitBreakerRule(
            name="volatility_circuit_breaker",
            signal="annualized_volatility",
            window=12,
            warning_threshold=0.0,
            reduce_only_threshold=0.0001,
            kill_switch_threshold=0.5,
            cooldown=timedelta(minutes=0),
            enabled=True,
        )
        state = CircuitBreakerState()
        state.register_rule(rule)

        logger = DummyLogger()
        marks = [Decimal(str(100 + (-1) ** i * 1.5)) for i in range(20)]
        outcome = runtime_check_volatility(
            symbol="BTC-USD",
            recent_marks=marks,
            rule=rule,
            state=state,
            now=lambda: dt.datetime.utcnow(),
            logger=logger,
        )
        assert outcome.triggered is True
        assert outcome.action is CircuitBreakerAction.REDUCE_ONLY
        snapshot = state.get(rule.name, "BTC-USD")
        assert snapshot is not None
        snapshots = state.snapshot()
        assert rule.name in snapshots and "BTC-USD" in snapshots[rule.name]

    def test_helper_respects_cooldown(self):
        rule = CircuitBreakerRule(
            name="volatility_circuit_breaker",
            signal="annualized_volatility",
            window=12,
            warning_threshold=0.0,
            reduce_only_threshold=0.0001,
            kill_switch_threshold=0.5,
            cooldown=timedelta(minutes=1),
            enabled=True,
        )
        state = CircuitBreakerState()
        state.register_rule(rule)
        state.record(
            rule.name,
            "BTC-USD",
            CircuitBreakerAction.WARNING,
            triggered_at=dt.datetime.utcnow(),
        )
        logger = DummyLogger()
        outcome = runtime_check_volatility(
            symbol="BTC-USD",
            recent_marks=[Decimal("100")] * 6,
            rule=rule,
            state=state,
            now=lambda: dt.datetime.utcnow(),
            logger=logger,
        )
        assert outcome.triggered is False
        snapshots = state.snapshot()
        assert rule.name in snapshots and "BTC-USD" in snapshots[rule.name]


def test_append_risk_metrics_records_snapshot():
    store = DummyEventStore()
    logger = DummyLogger()
    now = dt.datetime.utcnow()
    append_risk_metrics(
        event_store=store,
        now=lambda: now,
        equity=Decimal("1000"),
        positions={
            "BTC-USD": {"quantity": Decimal("0.1"), "mark": Decimal("40000")},
        },
        daily_pnl=Decimal("25"),
        start_of_day_equity=Decimal("900"),
        reduce_only=False,
        kill_switch_enabled=False,
        logger=logger,
    )
    assert store.metrics
    metric = store.metrics[0]["metrics"]
    assert metric["equity"] == "1000"
    assert Decimal(metric["exposure_pct"]) > 0


def test_check_correlation_risk_detects_concentration():
    events = []
    logger = DummyLogger()

    def log_event(event_type, details, *, guard):
        events.append((event_type, guard, details))

    result = check_correlation_risk(
        {
            "BTC": {"quantity": 1, "mark": 50000},
            "ETH": {"quantity": 0.1, "mark": 3000},
        },
        log_event=log_event,
        logger=logger,
    )
    assert result is True
    assert any(evt[0] == "concentration_risk" for evt in events)


# ===== Volatility Circuit Breaker Tests =====


def generate_marks_with_volatility(target_annual_vol: float, n: int = 30) -> list[Decimal]:
    """Generate synthetic price series with target annualized volatility.

    Uses alternating returns +/- s where s = target_vol / sqrt(252).
    """
    import math

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
    """Test volatility circuit breaker progressive warning/reduce-only/kill actions."""
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
    """Test circuit breaker cooldown prevents repeated triggers."""
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
