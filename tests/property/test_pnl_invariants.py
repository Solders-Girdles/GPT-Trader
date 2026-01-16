"""Property-based tests for PnL calculation invariants.

Tests critical PnL metric properties:
- Total PnL = realized + unrealized
- Win rate is in [0, 1] range
- Profit factor is non-negative
- Max drawdown calculations are consistent
- Equity change percentages are mathematically correct
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st

from gpt_trader.monitoring.daily_report.analytics import (
    calculate_pnl_metrics,
    calculate_trade_metrics,
)

# Strategies for generating valid PnL parameters
pnl_value_strategy = st.floats(
    min_value=-1000000, max_value=1000000, allow_nan=False, allow_infinity=False
)

positive_pnl_strategy = st.floats(
    min_value=0.01, max_value=1000000, allow_nan=False, allow_infinity=False
)

equity_strategy = st.floats(
    min_value=100, max_value=10000000, allow_nan=False, allow_infinity=False
)


def make_pnl_event(realized: float, unrealized: float) -> dict:
    """Create a PnL update event."""
    return {
        "type": "pnl_update",
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
    }


def make_fill_event(pnl: float, fee: float = 0.0) -> dict:
    """Create a fill event."""
    return {
        "type": "fill",
        "pnl": pnl,
        "fee": fee,
        "timestamp": "2025-01-01T12:00:00",
    }


def make_funding_event(amount: float) -> dict:
    """Create a funding payment event."""
    return {
        "type": "funding_payment",
        "amount": amount,
    }


@seed(4001)
@settings(max_examples=200, deadline=None)
@given(
    realized=pnl_value_strategy,
    unrealized=pnl_value_strategy,
    equity=equity_strategy,
)
def test_total_pnl_is_sum_of_components(
    realized: float,
    unrealized: float,
    equity: float,
) -> None:
    """
    Property: Total PnL = realized + unrealized.

    This fundamental relationship must always hold.
    """
    events = [make_pnl_event(realized, unrealized)]
    current_metrics = {"account": {"equity": equity}}

    metrics = calculate_pnl_metrics(events, current_metrics)

    expected_total = realized + unrealized
    tolerance = 0.01

    assert (
        abs(metrics["total_pnl"] - expected_total) < tolerance
    ), f"Total PnL {metrics['total_pnl']} != realized {realized} + unrealized {unrealized}"


@seed(4002)
@settings(max_examples=200, deadline=None)
@given(
    realized=pnl_value_strategy,
    unrealized=pnl_value_strategy,
    funding=pnl_value_strategy,
    equity=equity_strategy,
)
def test_pnl_components_correctly_extracted(
    realized: float,
    unrealized: float,
    funding: float,
    equity: float,
) -> None:
    """
    Property: PnL components should be correctly extracted from events.
    """
    events = [
        make_pnl_event(realized, unrealized),
        make_funding_event(funding),
    ]
    current_metrics = {"account": {"equity": equity}}

    metrics = calculate_pnl_metrics(events, current_metrics)

    tolerance = 0.01

    assert abs(metrics["realized_pnl"] - realized) < tolerance
    assert abs(metrics["unrealized_pnl"] - unrealized) < tolerance
    assert abs(metrics["funding_pnl"] - funding) < tolerance


@seed(4003)
@settings(max_examples=200, deadline=None)
@given(
    wins=st.lists(positive_pnl_strategy, min_size=0, max_size=20),
    losses=st.lists(positive_pnl_strategy, min_size=0, max_size=20),
)
def test_win_rate_bounded(
    wins: list[float],
    losses: list[float],
) -> None:
    """
    Property: Win rate should always be in [0, 1] range.
    """
    # Create fill events
    events = []
    for win in wins:
        events.append(make_fill_event(win))  # Positive PnL = win
    for loss in losses:
        events.append(make_fill_event(-loss))  # Negative PnL = loss

    if not events:
        return  # Skip empty case

    metrics = calculate_trade_metrics(events)

    assert 0 <= metrics["win_rate"] <= 1, f"Win rate {metrics['win_rate']} should be in [0, 1]"


@seed(4004)
@settings(max_examples=200, deadline=None)
@given(
    wins=st.lists(positive_pnl_strategy, min_size=0, max_size=20),
    losses=st.lists(positive_pnl_strategy, min_size=0, max_size=20),
)
def test_win_rate_calculation(
    wins: list[float],
    losses: list[float],
) -> None:
    """
    Property: Win rate = winning_trades / total_trades.
    """
    events = []
    for win in wins:
        events.append(make_fill_event(win))
    for loss in losses:
        events.append(make_fill_event(-loss))

    total = len(wins) + len(losses)

    if total == 0:
        return  # Skip empty case

    metrics = calculate_trade_metrics(events)

    expected_win_rate = len(wins) / total
    tolerance = 0.01

    assert (
        abs(metrics["win_rate"] - expected_win_rate) < tolerance
    ), f"Win rate {metrics['win_rate']} != expected {expected_win_rate}"


@seed(4005)
@settings(max_examples=200, deadline=None)
@given(
    wins=st.lists(positive_pnl_strategy, min_size=1, max_size=20),
    losses=st.lists(positive_pnl_strategy, min_size=1, max_size=20),
)
def test_profit_factor_non_negative(
    wins: list[float],
    losses: list[float],
) -> None:
    """
    Property: Profit factor should always be non-negative.

    profit_factor = gross_profit / gross_loss
    """
    events = []
    for win in wins:
        events.append(make_fill_event(win))
    for loss in losses:
        events.append(make_fill_event(-loss))

    metrics = calculate_trade_metrics(events)

    assert (
        metrics["profit_factor"] >= 0
    ), f"Profit factor {metrics['profit_factor']} should not be negative"


@seed(4006)
@settings(max_examples=200, deadline=None)
@given(
    wins=st.lists(positive_pnl_strategy, min_size=1, max_size=20),
    losses=st.lists(positive_pnl_strategy, min_size=1, max_size=20),
)
def test_profit_factor_calculation(
    wins: list[float],
    losses: list[float],
) -> None:
    """
    Property: Profit factor = gross_profit / gross_loss.
    """
    events = []
    for win in wins:
        events.append(make_fill_event(win))
    for loss in losses:
        events.append(make_fill_event(-loss))

    metrics = calculate_trade_metrics(events)

    gross_profit = sum(wins)
    gross_loss = sum(losses)

    if gross_loss > 0:
        expected_pf = gross_profit / gross_loss
        tolerance = 0.01

        assert (
            abs(metrics["profit_factor"] - expected_pf) < tolerance
        ), f"Profit factor {metrics['profit_factor']} != expected {expected_pf}"


@seed(4007)
@settings(max_examples=200, deadline=None)
@given(
    wins=st.lists(positive_pnl_strategy, min_size=0, max_size=20),
    losses=st.lists(positive_pnl_strategy, min_size=0, max_size=20),
)
def test_max_drawdown_non_negative(
    wins: list[float],
    losses: list[float],
) -> None:
    """
    Property: Max drawdown should always be non-negative.
    """
    events = []
    for win in wins:
        events.append(make_fill_event(win))
    for loss in losses:
        events.append(make_fill_event(-loss))

    if not events:
        return

    metrics = calculate_trade_metrics(events)

    assert (
        metrics["max_drawdown"] >= 0
    ), f"Max drawdown {metrics['max_drawdown']} should not be negative"


@seed(4008)
@settings(max_examples=200, deadline=None)
@given(
    wins=st.lists(positive_pnl_strategy, min_size=1, max_size=10),
    losses=st.lists(positive_pnl_strategy, min_size=1, max_size=10),
)
def test_trade_counts_correct(
    wins: list[float],
    losses: list[float],
) -> None:
    """
    Property: Trade counts should match input events.
    """
    events = []
    for win in wins:
        events.append(make_fill_event(win))
    for loss in losses:
        events.append(make_fill_event(-loss))

    metrics = calculate_trade_metrics(events)

    assert metrics["total_trades"] == len(wins) + len(losses)
    assert metrics["winning_trades"] == len(wins)
    assert metrics["losing_trades"] == len(losses)


@seed(4009)
@settings(max_examples=200, deadline=None)
@given(
    wins=st.lists(positive_pnl_strategy, min_size=1, max_size=10),
)
def test_all_wins_profit_factor_handling(
    wins: list[float],
) -> None:
    """
    Property: All wins with no losses should handle profit factor edge case.
    """
    events = [make_fill_event(win) for win in wins]

    metrics = calculate_trade_metrics(events)

    # Win rate should be 100%
    assert metrics["win_rate"] == 1.0
    # Profit factor should be 0 when gross_loss is 0 (by the implementation)
    # This is a design choice - could also be infinity
    assert metrics["profit_factor"] == 0


@seed(4010)
@settings(max_examples=200, deadline=None)
@given(
    losses=st.lists(positive_pnl_strategy, min_size=1, max_size=10),
)
def test_all_losses_win_rate_zero(
    losses: list[float],
) -> None:
    """
    Property: All losses should have 0% win rate.
    """
    events = [make_fill_event(-loss) for loss in losses]

    metrics = calculate_trade_metrics(events)

    assert metrics["win_rate"] == 0.0
    assert metrics["winning_trades"] == 0
    assert metrics["losing_trades"] == len(losses)


@seed(4011)
@settings(max_examples=150, deadline=None)
@given(
    wins=st.lists(positive_pnl_strategy, min_size=1, max_size=10),
    losses=st.lists(positive_pnl_strategy, min_size=1, max_size=10),
)
def test_average_win_loss_calculation(
    wins: list[float],
    losses: list[float],
) -> None:
    """
    Property: Average win/loss should be correctly calculated.
    """
    events = []
    for win in wins:
        events.append(make_fill_event(win))
    for loss in losses:
        events.append(make_fill_event(-loss))

    metrics = calculate_trade_metrics(events)

    expected_avg_win = sum(wins) / len(wins)
    expected_avg_loss = sum(losses) / len(losses)
    tolerance = 0.01

    assert (
        abs(metrics["avg_win"] - expected_avg_win) < tolerance
    ), f"Avg win {metrics['avg_win']} != expected {expected_avg_win}"
    assert (
        abs(metrics["avg_loss"] - expected_avg_loss) < tolerance
    ), f"Avg loss {metrics['avg_loss']} != expected {expected_avg_loss}"


@seed(4012)
@settings(max_examples=150, deadline=None)
@given(
    wins=st.lists(positive_pnl_strategy, min_size=1, max_size=10),
    losses=st.lists(positive_pnl_strategy, min_size=1, max_size=10),
)
def test_largest_win_loss_calculation(
    wins: list[float],
    losses: list[float],
) -> None:
    """
    Property: Largest win/loss should be the maximum values.
    """
    events = []
    for win in wins:
        events.append(make_fill_event(win))
    for loss in losses:
        events.append(make_fill_event(-loss))

    metrics = calculate_trade_metrics(events)

    expected_largest_win = max(wins)
    expected_largest_loss = max(losses)
    tolerance = 0.01

    assert (
        abs(metrics["largest_win"] - expected_largest_win) < tolerance
    ), f"Largest win {metrics['largest_win']} != expected {expected_largest_win}"
    assert (
        abs(metrics["largest_loss"] - expected_largest_loss) < tolerance
    ), f"Largest loss {metrics['largest_loss']} != expected {expected_largest_loss}"


@seed(4013)
@settings(max_examples=100, deadline=None)
@given(
    total_pnl=pnl_value_strategy,
    equity=equity_strategy,
)
def test_equity_change_percentage(
    total_pnl: float,
    equity: float,
) -> None:
    """
    Property: Equity change percentage should be correctly calculated.

    equity_change_pct = (total_pnl / prev_equity) * 100
    where prev_equity = equity - total_pnl
    """
    assume(abs(total_pnl) < equity)  # Avoid divide by zero

    events = [make_pnl_event(total_pnl, 0)]  # All realized for simplicity
    current_metrics = {"account": {"equity": equity}}

    metrics = calculate_pnl_metrics(events, current_metrics)

    prev_equity = equity - total_pnl
    if prev_equity > 0:
        expected_pct = (total_pnl / prev_equity) * 100
        tolerance = 0.1

        assert (
            abs(metrics["equity_change_pct"] - expected_pct) < tolerance
        ), f"Equity change % {metrics['equity_change_pct']} != expected {expected_pct}"


@seed(4014)
@settings(max_examples=100, deadline=None)
@given(
    fees=st.lists(
        st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
        min_size=0,
        max_size=20,
    ),
    equity=equity_strategy,
)
def test_fees_accumulation(
    fees: list[float],
    equity: float,
) -> None:
    """
    Property: Fees should be correctly accumulated from fill events.
    """
    events = [make_fill_event(0, fee) for fee in fees]
    current_metrics = {"account": {"equity": equity}}

    metrics = calculate_pnl_metrics(events, current_metrics)

    expected_fees = sum(fees)
    tolerance = 0.01

    assert (
        abs(metrics["fees_paid"] - expected_fees) < tolerance
    ), f"Fees paid {metrics['fees_paid']} != expected {expected_fees}"


@pytest.mark.property
class TestPnLPropertyBased:
    """Grouped property-based tests for PnL calculations."""

    def test_empty_events_returns_defaults(self) -> None:
        """Verify empty events return sensible defaults."""
        metrics = calculate_pnl_metrics([], {"account": {"equity": 10000}})

        assert metrics["realized_pnl"] == 0
        assert metrics["unrealized_pnl"] == 0
        assert metrics["funding_pnl"] == 0
        assert metrics["total_pnl"] == 0
        assert metrics["fees_paid"] == 0

    def test_empty_events_trade_metrics(self) -> None:
        """Verify empty events trade metrics return sensible defaults."""
        metrics = calculate_trade_metrics([])

        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0
        assert metrics["profit_factor"] == 0

    def test_sharpe_ratio_requires_variance(self) -> None:
        """Verify Sharpe ratio calculation handles edge cases."""
        # All same returns -> zero variance -> sharpe = 0
        events = [make_fill_event(100) for _ in range(5)]
        metrics = calculate_trade_metrics(events)

        # With all wins of same amount, std_dev should be 0
        # Implementation returns 0 when std_dev is 0
        assert metrics["sharpe_ratio"] == 0

    def test_max_drawdown_sequence(self) -> None:
        """Verify max drawdown calculation with known sequence."""
        # Sequence: +100, +100, -150, +50
        # Equity: 0 -> 100 -> 200 -> 50 -> 100
        # Peak: 200, Max drawdown: 200 - 50 = 150
        events = [
            make_fill_event(100),
            make_fill_event(100),
            make_fill_event(-150),
            make_fill_event(50),
        ]

        metrics = calculate_trade_metrics(events)

        assert metrics["max_drawdown"] == 150
