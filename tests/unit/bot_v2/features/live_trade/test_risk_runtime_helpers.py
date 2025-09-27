import datetime as dt
from decimal import Decimal

import pytest

from bot_v2.features.live_trade.risk_runtime import (
    append_risk_metrics,
    check_correlation_risk,
    check_mark_staleness,
    check_volatility_circuit_breaker,
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


def test_check_mark_staleness_triggers_event():
    last_update = {"BTC-USD": dt.datetime.utcnow() - dt.timedelta(seconds=5)}
    events = []

    def log_event(event_type, details, *, guard):
        events.append((event_type, details, guard))

    logger = DummyLogger()
    result = check_mark_staleness(
        symbol="BTC-USD",
        last_mark_update=last_update,
        now=lambda: dt.datetime.utcnow(),
        max_staleness_seconds=1,
        log_event=log_event,
        logger=logger,
    )
    assert result is True
    assert any(evt[0] == "stale_mark_price" for evt in events)
    assert logger.warnings  # guard emitted warning


def test_append_risk_metrics_records_snapshot():
    store = DummyEventStore()
    logger = DummyLogger()
    now = dt.datetime.utcnow()
    append_risk_metrics(
        event_store=store,
        now=lambda: now,
        equity=Decimal("1000"),
        positions={
            "BTC-USD": {"qty": Decimal("0.1"), "mark": Decimal("40000")},
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
            "BTC": {"qty": 1, "mark": 50000},
            "ETH": {"qty": 0.1, "mark": 3000},
        },
        log_event=log_event,
        logger=logger,
    )
    assert result is True
    assert any(evt[0] == "concentration_risk" for evt in events)


class _Config:
    enable_volatility_circuit_breaker = True
    volatility_window_periods = 12
    circuit_breaker_cooldown_minutes = 0
    volatility_warning_threshold = 0.0
    volatility_reduce_only_threshold = 0.0001
    volatility_kill_switch_threshold = 0.5
    kill_switch_enabled = False


def test_check_volatility_circuit_breaker_reduce_only():
    config = _Config()
    last_trigger = {}
    events = []

    def log_event(event_type, details, *, guard):
        events.append(event_type)

    actions = []

    def set_reduce_only(enabled, reason):
        actions.append((enabled, reason))

    logger = DummyLogger()
    # Create marks with sufficient volatility
    marks = [Decimal(str(100 + (-1) ** i * 1.5)) for i in range(20)]
    result = check_volatility_circuit_breaker(
        symbol="BTC-USD",
        recent_marks=marks,
        config=config,
        now=lambda: dt.datetime.utcnow(),
        last_trigger=last_trigger,
        set_reduce_only=set_reduce_only,
        log_event=log_event,
        logger=logger,
    )
    assert result["triggered"] is True
    assert actions and actions[0][0] is True
    assert "volatility_circuit_breaker" in events


def test_check_volatility_circuit_breaker_respects_cooldown():
    config = _Config()
    last_trigger = {"BTC-USD": dt.datetime.utcnow()}

    logger = DummyLogger()
    result = check_volatility_circuit_breaker(
        symbol="BTC-USD",
        recent_marks=[Decimal("100")] * 6,
        config=config,
        now=lambda: dt.datetime.utcnow(),
        last_trigger=last_trigger,
        set_reduce_only=lambda *args, **kwargs: None,
        log_event=lambda *args, **kwargs: None,
        logger=logger,
    )
    assert result["triggered"] is False
