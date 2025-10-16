from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot_v2.monitoring.domain.perps.margin import MarginWindow, MarginWindowPolicy


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


@pytest.fixture
def policy() -> MarginWindowPolicy:
    return MarginWindowPolicy()


def test_determine_current_window_boundaries(policy: MarginWindowPolicy):
    assert policy.determine_current_window(_utc(2025, 1, 1, 0, 5)) == MarginWindow.OVERNIGHT

    prefunding = policy.determine_current_window(_utc(2025, 1, 1, 23, 45))
    assert prefunding == MarginWindow.PRE_FUNDING

    intraday = policy.determine_current_window(_utc(2025, 1, 1, 14, 30))
    assert intraday == MarginWindow.INTRADAY

    normal = policy.determine_current_window(_utc(2025, 1, 1, 10, 0))
    assert normal == MarginWindow.NORMAL


def test_calculate_next_window_change_orders_transitions(policy: MarginWindowPolicy):
    start = _utc(2025, 1, 1, 21, 30)
    next_change = policy.calculate_next_window_change(start)
    assert next_change.hour == 22 and next_change.date() == start.date()

    after_late = _utc(2025, 1, 1, 23, 45)
    next_change = policy.calculate_next_window_change(after_late)
    assert next_change.hour == 0 and next_change.date() == (after_late + timedelta(days=1)).date()


@pytest.mark.parametrize(
    ("current", "next_window", "expect_reduce"),
    [
        (MarginWindow.NORMAL, MarginWindow.OVERNIGHT, True),
        (MarginWindow.NORMAL, MarginWindow.PRE_FUNDING, True),
        (MarginWindow.INTRADAY, MarginWindow.NORMAL, False),
        (MarginWindow.OVERNIGHT, MarginWindow.NORMAL, False),
    ],
)
def test_should_reduce_risk(policy: MarginWindowPolicy, current, next_window, expect_reduce):
    assert policy.should_reduce_risk(current, next_window) is expect_reduce
