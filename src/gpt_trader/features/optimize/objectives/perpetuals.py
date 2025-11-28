"""Perpetual futures-specific objective functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.backtesting.metrics.risk import RiskMetrics
    from gpt_trader.backtesting.metrics.statistics import TradeStatistics
    from gpt_trader.backtesting.types import BacktestResult


@dataclass
class FundingAdjustedReturnObjective:
    """
    Maximize return accounting for funding payments.

    For perpetual futures, funding payments can significantly impact returns.
    Positive funding PnL means you received funding payments (typically when
    holding positions opposite to the crowd), while negative means you paid
    funding.

    This objective uses the total return which already includes funding PnL,
    making it suitable for optimizing perpetual futures strategies.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "funding_adjusted_return"

    @property
    def direction(self) -> str:
        """Optimization direction."""
        return "maximize"

    def calculate(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> float:
        """
        Calculate funding-adjusted return.

        The total_return_usd already includes funding_pnl, so this
        returns the net effect of trading plus funding.
        """
        # total_return_usd includes realized_pnl + funding_pnl - fees
        return float(result.total_return_usd)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class FundingEfficiencyObjective:
    """
    Maximize trading PnL relative to funding costs.

    This measures how well trading profits offset funding costs.
    A high ratio indicates that trading profits significantly
    exceed any funding costs incurred.

    The objective is:
    - realized_pnl / abs(funding_pnl) when funding_pnl < 0 (paying funding)
    - realized_pnl when funding_pnl >= 0 (receiving funding is a bonus)

    Attributes:
        min_trades: Minimum trades required for feasibility
        min_funding_threshold: Minimum absolute funding to consider (avoids div by zero)
    """

    min_trades: int = 10
    min_funding_threshold: float = 0.01

    @property
    def name(self) -> str:
        """Objective name."""
        return "funding_efficiency"

    @property
    def direction(self) -> str:
        """Optimization direction."""
        return "maximize"

    def calculate(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> float:
        """
        Calculate funding efficiency ratio.

        Returns:
            - If funding is positive (received): just return realized PnL (funding is a bonus)
            - If funding is negative (paid): realized_pnl / abs(funding_pnl)
            - If funding is negligible: return realized PnL
        """
        realized_pnl = float(result.realized_pnl)
        funding_pnl = float(result.funding_pnl)

        # If we're receiving funding (positive), it's a bonus - just maximize trading PnL
        if funding_pnl >= 0:
            return realized_pnl

        # If funding cost is negligible, just return trading PnL
        abs_funding = abs(funding_pnl)
        if abs_funding < self.min_funding_threshold:
            return realized_pnl

        # Negative funding means we're paying - measure how well we offset it
        # High positive value means trading profits greatly exceed funding costs
        return realized_pnl / abs_funding

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades
