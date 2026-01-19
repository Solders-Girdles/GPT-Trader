"""Property-based tests for margin window policy invariants."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from gpt_trader.monitoring.domain.perps.margin import MarginWindow, MarginWindowPolicy
from tests.property.margin_invariants_test_helpers import hour_strategy, minute_strategy


@seed(2005)
@settings(max_examples=200, deadline=None)
@given(
    hour=hour_strategy,
    minute=minute_strategy,
)
def test_margin_window_deterministic(
    hour: int,
    minute: int,
) -> None:
    """Property: Same time should always produce same window."""
    policy = MarginWindowPolicy()
    test_time = datetime(2025, 1, 15, hour, minute, 0)

    window1 = policy.determine_current_window(test_time)
    window2 = policy.determine_current_window(test_time)

    assert window1 == window2, f"Window determination not deterministic for {test_time}"
    assert isinstance(window1, MarginWindow)


@seed(2006)
@settings(max_examples=100, deadline=None)
@given(
    hour=hour_strategy,
    minute=minute_strategy,
)
def test_margin_window_coverage(
    hour: int,
    minute: int,
) -> None:
    """Property: Every time maps to exactly one valid window."""
    policy = MarginWindowPolicy()
    test_time = datetime(2025, 1, 15, hour, minute, 0)

    window = policy.determine_current_window(test_time)

    valid_windows = {
        MarginWindow.NORMAL,
        MarginWindow.INTRADAY,
        MarginWindow.OVERNIGHT,
        MarginWindow.PRE_FUNDING,
    }

    assert window in valid_windows, f"Unknown window {window} for time {test_time}"


def test_margin_requirements_ordering() -> None:
    """Property: Window requirements maintain strictness ordering."""
    policy = MarginWindowPolicy()

    normal_req = policy.get_requirements(MarginWindow.NORMAL)
    intraday_req = policy.get_requirements(MarginWindow.INTRADAY)
    overnight_req = policy.get_requirements(MarginWindow.OVERNIGHT)
    pre_funding_req = policy.get_requirements(MarginWindow.PRE_FUNDING)

    assert normal_req.initial_rate <= intraday_req.initial_rate
    assert intraday_req.initial_rate <= overnight_req.initial_rate
    assert overnight_req.initial_rate <= pre_funding_req.initial_rate

    assert normal_req.max_leverage >= intraday_req.max_leverage
    assert intraday_req.max_leverage >= overnight_req.max_leverage
    assert overnight_req.max_leverage >= pre_funding_req.max_leverage


@seed(2010)
@settings(max_examples=100, deadline=None)
@given(
    window=st.sampled_from(list(MarginWindow)),
)
def test_margin_requirement_consistency(window: MarginWindow) -> None:
    """Property: Requirements are internally consistent."""
    policy = MarginWindowPolicy()
    req = policy.get_requirements(window)

    assert req.maintenance_rate < req.initial_rate

    expected_max_leverage = Decimal("1") / req.initial_rate
    tolerance = Decimal("0.1")
    assert (
        abs(req.max_leverage - expected_max_leverage) < tolerance
    ), f"Max leverage {req.max_leverage} inconsistent with initial rate {req.initial_rate}"


@pytest.mark.property
class TestMarginPropertyBased:
    """Grouped property-based tests for margin monitoring."""

    def test_margin_window_pre_funding_times(self) -> None:
        """Verify pre-funding window behavior around funding times."""
        policy = MarginWindowPolicy()

        test_time = datetime(2025, 1, 15, 23, 45, 0)
        window = policy.determine_current_window(test_time)
        assert window == MarginWindow.PRE_FUNDING

        test_time = datetime(2025, 1, 15, 7, 45, 0)
        window = policy.determine_current_window(test_time)
        assert window == MarginWindow.NORMAL

        test_time = datetime(2025, 1, 15, 8, 15, 0)
        window = policy.determine_current_window(test_time)
        assert window == MarginWindow.NORMAL

        test_time = datetime(2025, 1, 15, 15, 45, 0)
        window = policy.determine_current_window(test_time)
        assert window == MarginWindow.INTRADAY

    def test_overnight_window_hours(self) -> None:
        """Verify overnight window covers expected hours."""
        policy = MarginWindowPolicy()

        overnight_hours = [22, 23, 0, 1, 2, 3, 4, 5]

        for hour in overnight_hours:
            test_time = datetime(2025, 1, 15, hour, 30, 0)
            window = policy.determine_current_window(test_time)
            assert window in {
                MarginWindow.OVERNIGHT,
                MarginWindow.PRE_FUNDING,
            }, f"Hour {hour} should be OVERNIGHT or PRE_FUNDING, got {window}"
