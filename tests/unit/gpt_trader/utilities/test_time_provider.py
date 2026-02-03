"""Tests for clock/time provider utilities."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gpt_trader.utilities.time_provider import (
    FakeClock,
    SystemClock,
    get_clock,
    reset_clock,
    set_clock,
)


@pytest.fixture(autouse=True)
def _reset_clock_after_test() -> None:
    yield
    reset_clock()


def test_system_clock_returns_utc_and_time_values() -> None:
    clock = SystemClock()
    now = clock.now_utc()

    assert now.tzinfo == UTC
    assert abs((datetime.now(UTC) - now).total_seconds()) < 2.0
    assert isinstance(clock.time(), float)
    assert isinstance(clock.monotonic(), float)


def test_fake_clock_advances_deterministically() -> None:
    clock = FakeClock(start_time=100.0)

    assert clock.time() == 100.0
    assert clock.monotonic() == 100.0
    assert clock.now_utc() == datetime.fromtimestamp(100.0, UTC)

    clock.advance(5.5)

    assert clock.time() == 105.5
    assert clock.monotonic() == 105.5
    assert clock.now_utc() == datetime.fromtimestamp(105.5, UTC)


def test_fake_clock_set_datetime_updates_time() -> None:
    clock = FakeClock()
    target = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

    clock.set_datetime(target)

    assert clock.time() == target.timestamp()
    assert clock.now_utc() == target


def test_clock_helpers_switch_and_reset() -> None:
    default_clock = get_clock()
    fake = FakeClock(start_time=50.0)

    set_clock(fake)
    assert get_clock() is fake

    reset_clock()
    assert get_clock() is default_clock
