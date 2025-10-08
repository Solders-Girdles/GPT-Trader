from __future__ import annotations

from datetime import datetime, timedelta

from src.bot_v2.orchestration.market_monitor import MarketActivityMonitor


def test_market_monitor_handles_naive_timestamps() -> None:
    monitor = MarketActivityMonitor(symbols=["BTC-PERP"], max_staleness_ms=1000)
    # Simulate an update 5 seconds in the past using a naïve datetime
    monitor.last_update["BTC-PERP"] = datetime.utcnow() - timedelta(seconds=5)

    assert monitor.check_staleness() is True
