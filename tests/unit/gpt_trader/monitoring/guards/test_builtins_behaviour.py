from __future__ import annotations

from datetime import timedelta

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.guards.base import GuardConfig
from gpt_trader.monitoring.guards.builtins import (
    DailyLossGuard,
    ErrorRateGuard,
    PositionStuckGuard,
    StaleMarkGuard,
)
from gpt_trader.utilities.time_provider import FakeClock


def _config(name: str, *, threshold: float, window: int = 60) -> GuardConfig:
    return GuardConfig(
        name=name,
        threshold=threshold,
        window_seconds=window,
        severity=AlertSeverity.ERROR,
        cooldown_seconds=0,
    )


def test_daily_loss_guard_resets_each_day() -> None:
    clock = FakeClock()
    guard = DailyLossGuard(_config("daily_loss", threshold=100), time_provider=clock)

    first_alert = guard.check({"pnl": -60})
    assert first_alert is None
    assert guard.daily_pnl == -60

    clock.advance(timedelta(days=1, minutes=1).total_seconds())
    alert = guard.check({"pnl": -120})
    assert guard.daily_pnl == -120
    assert alert is not None
    assert "Daily loss limit breached" in alert.message


def test_stale_mark_guard_handles_formats() -> None:
    clock = FakeClock()
    guard = StaleMarkGuard(_config("stale_mark", threshold=30), time_provider=clock)

    fresh_time = clock.now() - timedelta(seconds=10)
    assert guard.check({"symbol": "BTC-PERP", "mark_timestamp": fresh_time.isoformat()}) is None

    stale_time = clock.now() - timedelta(seconds=90)
    alert = guard.check({"symbol": "BTC-PERP", "mark_timestamp": stale_time.isoformat()})
    assert alert is not None
    assert "Stale marks detected" in alert.message

    epoch_time = int((clock.now() - timedelta(seconds=120)).timestamp())
    second_alert = guard.check({"symbol": "ETH-PERP", "mark_timestamp": epoch_time})
    assert second_alert is not None


def test_error_rate_guard_sliding_window() -> None:
    clock = FakeClock()
    guard = ErrorRateGuard(_config("error_rate", threshold=3, window=10), time_provider=clock)

    for _ in range(3):
        assert guard.check({"error": True}) is None

    alert = guard.check({"error": True})
    assert alert is not None

    clock.advance(11)
    guard.check({"error": False})
    assert len(guard.error_times) == 0


def test_position_stuck_guard_tracks_and_clears() -> None:
    clock = FakeClock()
    guard = PositionStuckGuard(_config("stuck", threshold=60), time_provider=clock)

    context = {"positions": {"BTC-PERP": {"quantity": 1}}}
    assert guard.check(context) is None

    clock.advance(70)
    alert = guard.check(context)
    assert alert is not None
    assert "Stuck positions detected" in alert.message

    guard.check({"positions": {"BTC-PERP": {"quantity": 0}}})
    assert "BTC-PERP" not in guard.position_times
