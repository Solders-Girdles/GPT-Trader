"""Simulation utilities for Kelly-based sizing."""

from __future__ import annotations

from bot_v2.features.position_sizing.types import TradeStatistics

from .calculations import fractional_kelly, kelly_criterion


def simulate_kelly_growth(
    returns: list[float],
    kelly_fraction: float,
    initial_wealth: float = 1.0,
) -> float:
    """Simulate wealth growth using Kelly sizing on historical returns."""
    wealth = initial_wealth

    for trade_return in returns:
        position_size = kelly_fraction * wealth
        wealth += position_size * trade_return
        wealth = max(0.01, wealth)

    return wealth


def optimal_kelly_fraction(
    returns: list[float],
    test_fractions: list[float] | None = None,
) -> tuple[float, float]:
    """Find the fractional Kelly size that maximizes wealth on sample data."""
    if not returns or len(returns) < 10:
        return 0.0, 1.0

    if test_fractions is None:
        test_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

    stats = TradeStatistics.from_returns(returns)
    full_kelly = kelly_criterion(stats.win_rate, stats.avg_win_return, stats.avg_loss_return)

    if full_kelly <= 0:
        return 0.0, 1.0

    best_fraction = 0.0
    best_wealth = 1.0

    for fraction in test_fractions:
        kelly_size = full_kelly * fraction
        wealth = simulate_kelly_growth(returns, kelly_size)
        if wealth > best_wealth:
            best_wealth = wealth
            best_fraction = fraction

    return best_fraction, best_wealth


__all__ = ["simulate_kelly_growth", "optimal_kelly_fraction"]
