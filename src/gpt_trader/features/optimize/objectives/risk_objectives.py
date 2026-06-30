"""Risk-focused single-metric objective functions for optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.backtesting.metrics.risk import RiskMetrics
    from gpt_trader.backtesting.metrics.statistics import TradeStatistics
    from gpt_trader.backtesting.types import BacktestResult


@dataclass
class MaxDrawdownObjective:
    """
    Minimize maximum drawdown percentage.

    Useful when capital preservation is the primary goal.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 5

    @property
    def name(self) -> str:
        """Objective name."""
        return "max_drawdown"

    @property
    def direction(self) -> str:
        """Optimization direction."""
        return "minimize"

    def calculate(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> float:
        """Calculate max drawdown objective value (to minimize)."""
        return float(risk_metrics.max_drawdown_pct)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


# =============================================================================
# Risk-Focused Objectives
# =============================================================================


@dataclass
class ValueAtRisk95Objective:
    """
    Minimize 95% daily Value at Risk.

    VaR measures the potential loss at a given confidence level.
    Lower VaR indicates a strategy with more predictable downside risk.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "var_95_daily"

    @property
    def direction(self) -> str:
        """Optimization direction."""
        return "minimize"

    def calculate(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> float:
        """Calculate 95% VaR objective value (to minimize)."""
        return float(risk_metrics.var_95_daily)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class ValueAtRisk99Objective:
    """
    Minimize 99% daily Value at Risk.

    VaR99 measures extreme tail risk at the 99% confidence level.
    More conservative than VaR95, focusing on worst-case scenarios.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "var_99_daily"

    @property
    def direction(self) -> str:
        """Optimization direction."""
        return "minimize"

    def calculate(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> float:
        """Calculate 99% VaR objective value (to minimize)."""
        return float(risk_metrics.var_99_daily)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class DrawdownRecoveryObjective:
    """
    Minimize drawdown duration (time to recover from peak).

    Strategies that recover quickly from losses are preferred.
    Long drawdown periods indicate difficulty recovering from losses.

    Attributes:
        min_trades: Minimum trades required for feasibility
        max_duration_days: Maximum allowed drawdown duration for feasibility
    """

    min_trades: int = 10
    max_duration_days: int = 30

    @property
    def name(self) -> str:
        """Objective name."""
        return "drawdown_recovery"

    @property
    def direction(self) -> str:
        """Optimization direction."""
        return "minimize"

    def calculate(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> float:
        """Calculate drawdown duration objective value (to minimize)."""
        return float(risk_metrics.drawdown_duration_days)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if trade count and duration constraints are met."""
        return (
            trade_statistics.total_trades >= self.min_trades
            and risk_metrics.drawdown_duration_days <= self.max_duration_days
        )


@dataclass
class DownsideVolatilityObjective:
    """
    Minimize downside volatility (negative return variance only).

    Unlike total volatility, downside volatility only considers negative returns.
    Lower values indicate more stable negative return distribution.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "downside_volatility"

    @property
    def direction(self) -> str:
        """Optimization direction."""
        return "minimize"

    def calculate(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> float:
        """Calculate downside volatility objective value (to minimize)."""
        return float(risk_metrics.downside_volatility)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class TailRiskAdjustedReturnObjective:
    """
    Maximize return per unit of tail risk (return / VaR99).

    Similar to Sharpe ratio but uses tail risk instead of volatility.
    Higher values indicate better returns relative to extreme loss potential.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "tail_risk_adjusted_return"

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
        """Calculate tail-risk-adjusted return."""
        var_99 = float(risk_metrics.var_99_daily)
        if var_99 <= 0:
            return float("-inf")
        return float(risk_metrics.total_return_pct) / var_99

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


# =============================================================================
# Trade Quality Objectives
# =============================================================================
