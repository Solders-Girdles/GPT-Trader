"""Base protocol for objective functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from gpt_trader.backtesting.metrics.risk import RiskMetrics
    from gpt_trader.backtesting.metrics.statistics import TradeStatistics
    from gpt_trader.backtesting.types import BacktestResult


@runtime_checkable
class ObjectiveFunction(Protocol):
    """
    Protocol for optimization objective functions.

    Objective functions evaluate backtest results and return a single scalar
    value to optimize. They can also define feasibility constraints.

    Attributes:
        name: Human-readable name for the objective
        direction: Optimization direction ("maximize" or "minimize")
    """

    @property
    def name(self) -> str:
        """Human-readable name for this objective."""
        ...

    @property
    def direction(self) -> str:
        """Optimization direction: 'maximize' or 'minimize'."""
        ...

    def calculate(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> float:
        """
        Calculate the objective value from backtest results.

        Args:
            result: BacktestResult from the backtest run
            risk_metrics: RiskMetrics calculated from the broker
            trade_statistics: TradeStatistics calculated from the broker

        Returns:
            Scalar objective value to optimize
        """
        ...

    def is_feasible(
        self,
        result: BacktestResult,
        risk_metrics: RiskMetrics,
        trade_statistics: TradeStatistics,
    ) -> bool:
        """
        Check if the result meets feasibility constraints.

        Infeasible trials receive worst-case objective values.

        Args:
            result: BacktestResult from the backtest run
            risk_metrics: RiskMetrics calculated from the broker
            trade_statistics: TradeStatistics calculated from the broker

        Returns:
            True if constraints are satisfied, False otherwise
        """
        ...
