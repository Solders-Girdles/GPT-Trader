"""Deterministic time tests for StatusReporter."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.utilities.time_provider import FakeClock, reset_clock, set_clock


@pytest.fixture
def fake_clock() -> FakeClock:
    clock = FakeClock(start_time=1_700_000_000.0)
    set_clock(clock)
    yield clock
    reset_clock()


def test_status_reporter_uses_fake_clock_for_timestamps(fake_clock: FakeClock) -> None:
    reporter = StatusReporter()
    reporter._start_time = fake_clock.time()

    status = reporter.get_status()
    expected_iso = datetime.fromtimestamp(fake_clock.time(), UTC).isoformat().replace("+00:00", "Z")

    assert status.timestamp == fake_clock.time()
    assert status.timestamp_iso == expected_iso
    assert status.engine.uptime_seconds == 0.0

    fake_clock.advance(12.5)
    reporter.record_cycle()

    status = reporter.get_status()
    assert status.timestamp == fake_clock.time()
    assert status.engine.uptime_seconds == 12.5
    assert status.engine.last_cycle_time == fake_clock.time()
