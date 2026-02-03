"""Tests for clock abstractions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gpt_trader.utilities.time_provider import FakeClock


def test_fake_clock_set_time_updates_now() -> None:
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    clock = FakeClock(base_time)

    assert clock.now() == base_time

    new_time = datetime(2024, 1, 1, 0, 1, 0, tzinfo=UTC)
    clock.set_time(new_time)

    assert clock.now() == new_time


def test_fake_clock_advance_updates_time() -> None:
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    clock = FakeClock(base_time)

    clock.advance(90.5)

    expected = base_time + timedelta(seconds=90.5)
    assert clock.now() == expected
    assert clock.time() == expected.timestamp()
