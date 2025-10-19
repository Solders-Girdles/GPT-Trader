import datetime as dt

from bot_v2.orchestration.market_monitor import MarketActivityMonitor
from bot_v2.orchestration.session_guard import TradingSessionGuard


def _make_time(hour, minute=0):
    return dt.time(hour=hour, minute=minute)


def test_guard_allows_when_not_configured():
    guard = TradingSessionGuard(start=None, end=None, trading_days=None)
    assert guard.should_trade() is True


def test_guard_enforces_day_and_window():
    guard = TradingSessionGuard(
        start=_make_time(9, 0),
        end=_make_time(17, 0),
        trading_days=["Monday", "Tuesday"],
    )
    now = dt.datetime(2025, 1, 6, 10, 0, 0)  # Monday
    assert guard.should_trade(now)
    late = dt.datetime(2025, 1, 6, 18, 0, 0)
    assert guard.should_trade(late) is False
    wed = dt.datetime(2025, 1, 8, 10, 0, 0)
    assert guard.should_trade(wed) is False


def test_guard_handles_overnight_window():
    guard = TradingSessionGuard(
        start=_make_time(22, 0),
        end=_make_time(6, 0),
        trading_days=["Monday", "Tuesday"],
    )
    before_midnight = dt.datetime(2025, 1, 6, 23, 0, 0)
    after_midnight = dt.datetime(2025, 1, 7, 1, 0, 0)
    assert guard.should_trade(before_midnight)
    assert guard.should_trade(after_midnight)


class _HeartbeatRecorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **payload):
        self.calls.append(payload)


def test_market_monitor_updates_and_staleness():
    recorder = _HeartbeatRecorder()
    monitor = MarketActivityMonitor(["BTC-USD"], max_staleness_ms=100, heartbeat_logger=recorder)

    now = dt.datetime.utcnow()
    monitor.last_update["BTC-USD"] = now - dt.timedelta(milliseconds=50)
    assert monitor.check_staleness() is False
    monitor.last_update["BTC-USD"] = now - dt.timedelta(milliseconds=200)
    assert monitor.check_staleness() is True
    assert any(call.get("source") == "staleness_guard" for call in recorder.calls)


def test_market_monitor_fallback_to_rest():
    monitor = MarketActivityMonitor(["BTC-USD"], max_failures=0, max_staleness_ms=10)
    assert monitor.should_fallback_to_rest() is False
    monitor.record_failure()
    assert monitor.should_fallback_to_rest() is True
