"""Single-metric objective functions for optimization."""

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
class WinRateObjective:
    """
    Maximize win rate percentage.

    Simple objective that maximizes the percentage of winning trades.
    Best combined with constraints on trade frequency.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 20

    @property
    def name(self) -> str:
        """Objective name."""
        return "win_rate"

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
        """Calculate win rate objective value."""
        return float(trade_statistics.win_rate)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class ProfitFactorObjective:
    """
    Maximize profit factor (gross profit / gross loss).

    A profit factor > 1 indicates a profitable strategy.
    Values above 2.0 are considered very good.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "profit_factor"

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
        """Calculate profit factor objective value."""
        return float(trade_statistics.profit_factor)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


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


@dataclass
class StreakConsistencyObjective:
    """
    Minimize maximum consecutive losses.

    Strategies with long losing streaks are psychologically harder to trade
    and may indicate fragility in the underlying edge.

    Attributes:
        min_trades: Minimum trades required for feasibility
        max_allowed_consecutive_losses: Maximum streak for feasibility
    """

    min_trades: int = 20
    max_allowed_consecutive_losses: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "streak_consistency"

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
        """Calculate max consecutive losses (to minimize)."""
        return float(trade_statistics.max_consecutive_losses)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check trade count and streak constraints."""
        return (
            trade_statistics.total_trades >= self.min_trades
            and trade_statistics.max_consecutive_losses <= self.max_allowed_consecutive_losses
        )


@dataclass
class CostAdjustedReturnObjective:
    """
    Maximize net return after trading costs (fees).

    Penalizes strategies with high execution costs.
    Uses realized PnL which accounts for fees paid.

    Attributes:
        min_trades: Minimum trades required for feasibility
    """

    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "cost_adjusted_return"

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
        """Calculate cost-adjusted return (realized PnL after fees)."""
        # realized_pnl already accounts for fees in most implementations
        # but we can also compute: total_return_usd - fees_paid for net
        return float(result.total_return_usd - result.fees_paid)

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class ExecutionQualityObjective:
    """
    Maximize execution quality (low slippage + high fill rate).

    Composite metric useful for strategies dependent on precise execution,
    such as market-making or limit-order strategies.

    Attributes:
        min_trades: Minimum trades required for feasibility
        slippage_weight: Weight for slippage component (0-1)
        fill_rate_weight: Weight for fill rate component (0-1)
    """

    min_trades: int = 10
    slippage_weight: float = 0.5
    fill_rate_weight: float = 0.5

    @property
    def name(self) -> str:
        """Objective name."""
        return "execution_quality"

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
        """Calculate execution quality score."""
        # Normalize slippage (lower is better, invert for maximize)
        # Assuming max slippage of 100 bps as a reference
        slippage_score = max(0.0, 100.0 - float(trade_statistics.avg_slippage_bps))
        fill_rate_score = float(trade_statistics.limit_fill_rate)

        return self.slippage_weight * slippage_score + self.fill_rate_weight * fill_rate_score

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


# =============================================================================
# Exposure/Timing Objectives
# =============================================================================


@dataclass
class TimeEfficiencyObjective:
    """
    Maximize return per unit of time in market.

    Rewards strategies that achieve returns with minimal market exposure.
    Higher values indicate efficient capital deployment.

    Attributes:
        min_trades: Minimum trades required for feasibility
        min_time_in_market_pct: Minimum exposure to avoid division issues
    """

    min_trades: int = 10
    min_time_in_market_pct: float = 5.0

    @property
    def name(self) -> str:
        """Objective name."""
        return "time_efficiency"

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
        """Calculate return per unit of time in market."""
        time_in_market = float(risk_metrics.time_in_market_pct)
        if time_in_market < self.min_time_in_market_pct:
            return float("-inf")  # Too little exposure to be meaningful
        return float(risk_metrics.total_return_pct) / time_in_market

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """Check if minimum trade count is met."""
        return trade_statistics.total_trades >= self.min_trades


@dataclass
class HoldDurationObjective:
    """
    Minimize deviation from target hold duration.

    Optimizes for strategies with desired holding periods.
    Useful when targeting specific trading frequencies.

    Attributes:
        target_minutes: Target average hold time in minutes
        min_trades: Minimum trades required for feasibility
    """

    target_minutes: float = 60.0  # Default: 1-hour target
    min_trades: int = 10

    @property
    def name(self) -> str:
        """Objective name."""
        return "hold_duration"

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
        """Calculate absolute deviation from target hold time."""
        avg_hold = float(trade_statistics.avg_hold_time_minutes)
        return abs(avg_hold - self.target_minutes)

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
