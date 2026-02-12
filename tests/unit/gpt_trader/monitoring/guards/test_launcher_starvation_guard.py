from __future__ import annotations

from datetime import datetime

from gpt_trader.monitoring.guards import GuardConfig, GuardStatus, LauncherStarvationGuard
from gpt_trader.utilities.time_provider import FakeClock


def _create_guard(
    *, threshold: int = 2, cooldown_seconds: int = 30, time_provider=None
) -> LauncherStarvationGuard:
    config = GuardConfig(
        name="launcher_starvation",
        threshold=threshold,
        cooldown_seconds=cooldown_seconds,
    )
    return LauncherStarvationGuard(config, time_provider=time_provider)


def test_alert_triggers_when_streak_exceeds_threshold() -> None:
    guard = _create_guard()
    alert = guard.check(
        {
            "no_candidate_streak": 4,
            "opportunity_id": "opp_17b60321e9",
        }
    )

    assert alert is not None
    assert "opp_17b60321e9" in alert.message
    assert "4 consecutive" in alert.message or "4" in alert.message
    assert guard.status == GuardStatus.BREACHED


def test_no_alert_when_streak_below_threshold() -> None:
    guard = _create_guard(threshold=5)
    assert guard.check({"no_candidate_streak": 1}) is None
    assert guard.status == GuardStatus.HEALTHY


def test_cooldown_prevents_alerts_until_elapsed() -> None:
    clock = FakeClock(start_datetime=datetime(2025, 1, 1))
    guard = _create_guard(cooldown_seconds=60, time_provider=clock)

    first_alert = guard.check({"no_candidate_streak": 5})
    assert first_alert is not None

    clock.advance(30)
    assert guard.check({"no_candidate_streak": 6}) is None

    clock.advance(31)
    second_alert = guard.check({"no_candidate_streak": 7})
    assert second_alert is not None
