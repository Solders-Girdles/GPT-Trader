"""Return-focused single-metric objective functions for optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.backtesting.metrics.risk import RiskMetrics
    from gpt_trader.backtesting.metrics.statistics import TradeStatistics
    from gpt_trader.backtesting.types import BacktestResult


@dataclass
class SharpeRatioObjective:
    """
    Maximize risk-adjusted returns via Sharpe ratio.

    The Sharpe ratio measures excess return per unit of risk (volatility).
    Higher values indicate better risk-adjusted performance.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "sharpe_ratio"

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
        """Calculate Sharpe ratio objective value."""
        if risk_metrics.sharpe_ratio is None:
            return float("-inf")
        return float(risk_metrics.sharpe_ratio)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class SortinoRatioObjective:
    """
    Maximize downside risk-adjusted returns via Sortino ratio.

    The Sortino ratio is similar to Sharpe but only penalizes downside volatility,
    not overall volatility. This is preferred when upside variance is desirable.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "sortino_ratio"

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
        """Calculate Sortino ratio objective value."""
        if risk_metrics.sortino_ratio is None:
            return float("-inf")
        return float(risk_metrics.sortino_ratio)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class TotalReturnObjective:
    """
    Maximize total return percentage.

    Simple objective that maximizes raw returns without risk adjustment.
    Best used with additional constraints on drawdown.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 5

    @property
    def name(self) -> str:
        """Objective name."""
        return "total_return"

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
        """Calculate total return objective value."""
        return float(result.total_return)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class CalmarRatioObjective:
    """
    Maximize return per unit of maximum drawdown via Calmar ratio.

    The Calmar ratio is annual return divided by maximum drawdown.
    It focuses on worst-case risk rather than volatility.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "calmar_ratio"

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
        """Calculate Calmar ratio objective value."""
        if risk_metrics.calmar_ratio is None:
            return float("-inf")
        return float(risk_metrics.calmar_ratio)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class LeverageAdjustedReturnObjective:
    """
    Maximize return divided by average leverage used.

    Rewards strategies that achieve returns with minimal leverage.
    Higher values indicate efficient use of margin.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "leverage_adjusted_return"

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
        """Calculate return per unit of leverage."""
        avg_leverage = float(risk_metrics.avg_leverage_used)
        if avg_leverage <= 0:
            avg_leverage = 1.0  # Default to 1x if not tracked
        return float(risk_metrics.total_return_pct) / avg_leverage

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades
