from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from scripts.ops import canary_watchdog


def _make_event(event_type: str, age: int) -> canary_watchdog.liveness_check.EventAge:
    return canary_watchdog.liveness_check.EventAge(
        event_type=event_type,
        last_ts="-",
        age_seconds=age,
        event_id=None,
    )


def test_green_does_not_restart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _green(
        *_: object, **__: object
    ) -> tuple[list[canary_watchdog.liveness_check.EventAge], bool]:
        rows = [_make_event("heartbeat", 10)]
        return rows, True

    restart_calls: list[int] = []

    def _restart_canary(**_: object) -> int:
        restart_calls.append(1)
        return 0

    monkeypatch.setattr(canary_watchdog.liveness_check, "check_liveness", _green)
    monkeypatch.setattr(canary_watchdog.canary_process, "restart_canary", _restart_canary)

    state = canary_watchdog.WatchdogState()
    outcome = canary_watchdog._poll_once(
        runtime_root=tmp_path,
        profile="canary",
        event_types=["heartbeat"],
        max_age_seconds=300,
        auto_restart=True,
        restart_after_reds=2,
        restart_cooldown_seconds=900,
        state=state,
    )

    assert outcome.is_green is True
    assert outcome.restart_failed is False
    assert restart_calls == []


def test_red_twice_triggers_restart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reds = iter([False, False])

    def _red(
        *_: object, **__: object
    ) -> tuple[list[canary_watchdog.liveness_check.EventAge], bool]:
        rows = [_make_event("price_tick", 600)]
        return rows, next(reds)

    restart_calls: list[int] = []

    def _restart_canary(**_: object) -> int:
        restart_calls.append(1)
        return 0

    monkeypatch.setattr(canary_watchdog.liveness_check, "check_liveness", _red)
    monkeypatch.setattr(canary_watchdog.canary_process, "restart_canary", _restart_canary)

    state = canary_watchdog.WatchdogState()
    canary_watchdog._poll_once(
        runtime_root=tmp_path,
        profile="canary",
        event_types=["price_tick"],
        max_age_seconds=300,
        auto_restart=True,
        restart_after_reds=2,
        restart_cooldown_seconds=900,
        state=state,
    )
    outcome = canary_watchdog._poll_once(
        runtime_root=tmp_path,
        profile="canary",
        event_types=["price_tick"],
        max_age_seconds=300,
        auto_restart=True,
        restart_after_reds=2,
        restart_cooldown_seconds=900,
        state=state,
    )

    assert outcome.decision == "restart_triggered"
    assert restart_calls == [1]


def test_cooldown_prevents_restart(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _red(
        *_: object, **__: object
    ) -> tuple[list[canary_watchdog.liveness_check.EventAge], bool]:
        rows = [_make_event("heartbeat", 800)]
        return rows, False

    restart_calls: list[int] = []

    def _restart_canary(**_: object) -> int:
        restart_calls.append(1)
        return 0

    monkeypatch.setattr(canary_watchdog.liveness_check, "check_liveness", _red)
    monkeypatch.setattr(canary_watchdog.canary_process, "restart_canary", _restart_canary)

    state = canary_watchdog.WatchdogState(consecutive_reds=1, last_restart_ts=100.0)

    monkeypatch.setattr(canary_watchdog, "_now", lambda: 105.0)

    outcome = canary_watchdog._poll_once(
        runtime_root=tmp_path,
        profile="canary",
        event_types=["heartbeat"],
        max_age_seconds=300,
        auto_restart=True,
        restart_after_reds=2,
        restart_cooldown_seconds=900,
        state=state,
    )

    assert outcome.decision.startswith("cooldown_remaining=")
    assert restart_calls == []


def test_once_exit_codes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _green(
        *_: object, **__: object
    ) -> tuple[list[canary_watchdog.liveness_check.EventAge], bool]:
        rows = [_make_event("heartbeat", 10)]
        return rows, True

    def _red(
        *_: object, **__: object
    ) -> tuple[list[canary_watchdog.liveness_check.EventAge], bool]:
        rows = [_make_event("heartbeat", 600)]
        return rows, False

    args = argparse.Namespace(
        profile="canary",
        runtime_root=tmp_path,
        event_type=None,
        max_age_seconds=300,
        poll_seconds=60,
        restart_after_reds=2,
        restart_cooldown_seconds=900,
        auto_restart=False,
        once=True,
    )

    monkeypatch.setattr(canary_watchdog.liveness_check, "check_liveness", _green)
    monkeypatch.setattr(canary_watchdog, "_parse_args", lambda: args)
    assert canary_watchdog.main() == 0

    monkeypatch.setattr(canary_watchdog.liveness_check, "check_liveness", _red)
    assert canary_watchdog.main() == 1


def test_once_persists_reds_for_auto_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _red(
        *_: object, **__: object
    ) -> tuple[list[canary_watchdog.liveness_check.EventAge], bool]:
        rows = [_make_event("heartbeat", 600)]
        return rows, False

    restart_calls: list[int] = []

    def _restart_canary(**_: object) -> int:
        restart_calls.append(1)
        return 0

    args = argparse.Namespace(
        profile="canary",
        runtime_root=tmp_path,
        event_type=None,
        max_age_seconds=300,
        poll_seconds=60,
        restart_after_reds=2,
        restart_cooldown_seconds=900,
        auto_restart=True,
        once=True,
    )

    monkeypatch.setattr(canary_watchdog.liveness_check, "check_liveness", _red)
    monkeypatch.setattr(canary_watchdog.canary_process, "restart_canary", _restart_canary)
    monkeypatch.setattr(canary_watchdog, "_parse_args", lambda: args)

    monkeypatch.setattr(canary_watchdog, "_now", lambda: 100.0)
    assert canary_watchdog.main() == 1
    assert restart_calls == []

    monkeypatch.setattr(canary_watchdog, "_now", lambda: 200.0)
    assert canary_watchdog.main() == 1
    assert restart_calls == [1]


def test_once_green_resets_persisted_reds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    restart_calls: list[int] = []

    def _restart_canary(**_: object) -> int:
        restart_calls.append(1)
        return 0

    args = argparse.Namespace(
        profile="canary",
        runtime_root=tmp_path,
        event_type=None,
        max_age_seconds=300,
        poll_seconds=60,
        restart_after_reds=2,
        restart_cooldown_seconds=900,
        auto_restart=True,
        once=True,
    )

    monkeypatch.setattr(canary_watchdog.canary_process, "restart_canary", _restart_canary)
    monkeypatch.setattr(canary_watchdog, "_parse_args", lambda: args)

    def _set_liveness(is_green: bool, age: int) -> None:
        def _check(
            *_: object, **__: object
        ) -> tuple[list[canary_watchdog.liveness_check.EventAge], bool]:
            rows = [_make_event("heartbeat", age)]
            return rows, is_green

        monkeypatch.setattr(canary_watchdog.liveness_check, "check_liveness", _check)

    _set_liveness(False, 600)
    monkeypatch.setattr(canary_watchdog, "_now", lambda: 100.0)
    assert canary_watchdog.main() == 1

    _set_liveness(True, 10)
    monkeypatch.setattr(canary_watchdog, "_now", lambda: 160.0)
    assert canary_watchdog.main() == 0

    _set_liveness(False, 600)
    monkeypatch.setattr(canary_watchdog, "_now", lambda: 220.0)
    assert canary_watchdog.main() == 1
    assert restart_calls == []

    _set_liveness(False, 600)
    monkeypatch.setattr(canary_watchdog, "_now", lambda: 280.0)
    assert canary_watchdog.main() == 1
    assert restart_calls == [1]
