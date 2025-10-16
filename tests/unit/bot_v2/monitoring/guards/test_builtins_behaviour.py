from __future__ import annotations

from datetime import datetime, timedelta

from bot_v2.monitoring.alert_types import AlertSeverity
from bot_v2.monitoring.guards.base import GuardConfig
from bot_v2.monitoring.guards.builtins import (
    DailyLossGuard,
    ErrorRateGuard,
    PositionStuckGuard,
    StaleMarkGuard,
)


def _config(name: str, *, threshold: float, window: int = 60) -> GuardConfig:
    return GuardConfig(
        name=name,
        threshold=threshold,
        window_seconds=window,
        severity=AlertSeverity.ERROR,
        cooldown_seconds=0,
    )


def test_daily_loss_guard_resets_each_day(frozen_time):
    guard = DailyLossGuard(_config("daily_loss", threshold=100))

    first_alert = guard.check({"pnl": -60})
    assert first_alert is None
    assert guard.daily_pnl == -60

    frozen_time.tick(delta=timedelta(days=1, minutes=1))
    alert = guard.check({"pnl": -120})
    assert guard.daily_pnl == -120
    assert alert is not None
    assert "Daily loss limit breached" in alert.message


def test_stale_mark_guard_handles_formats(frozen_time):
    guard = StaleMarkGuard(_config("stale_mark", threshold=30))

    fresh_time = datetime.now() - timedelta(seconds=10)
    assert guard.check({"symbol": "BTC-PERP", "mark_timestamp": fresh_time.isoformat()}) is None

    stale_time = datetime.now() - timedelta(seconds=90)
    alert = guard.check({"symbol": "BTC-PERP", "mark_timestamp": stale_time.isoformat()})
    assert alert is not None
    assert "Stale marks detected" in alert.message

    epoch_time = int((datetime.now() - timedelta(seconds=120)).timestamp())
    second_alert = guard.check({"symbol": "ETH-PERP", "mark_timestamp": epoch_time})
    assert second_alert is not None


def test_error_rate_guard_sliding_window(frozen_time):
    guard = ErrorRateGuard(_config("error_rate", threshold=3, window=10))

    for _ in range(3):
        assert guard.check({"error": True}) is None

    alert = guard.check({"error": True})
    assert alert is not None

    frozen_time.tick(delta=timedelta(seconds=11))
    guard.check({"error": False})
    assert len(guard.error_times) == 0


def test_position_stuck_guard_tracks_and_clears(frozen_time):
    guard = PositionStuckGuard(_config("stuck", threshold=60))

    context = {"positions": {"BTC-PERP": {"quantity": 1}}}
    assert guard.check(context) is None

    frozen_time.tick(delta=timedelta(seconds=70))
    alert = guard.check(context)
    assert alert is not None
    assert "Stuck positions detected" in alert.message

    guard.check({"positions": {"BTC-PERP": {"quantity": 0}}})
    assert "BTC-PERP" not in guard.position_times
