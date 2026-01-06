"""
Performance metrics calculation for backtest evaluation.

Provides comprehensive risk-adjusted performance statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpt_trader.features.research.backtesting.simulator import BacktestResult


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a backtest.

    Attributes:
        total_return: Total return as decimal (0.1 = 10%).
        annualized_return: Annualized return.
        sharpe_ratio: Risk-adjusted return (Sharpe ratio).
        sortino_ratio: Downside risk-adjusted return.
        max_drawdown: Maximum peak-to-trough decline.
        max_drawdown_duration: Longest drawdown period.
        win_rate: Percentage of winning trades.
        profit_factor: Gross profit / gross loss.
        avg_trade_return: Average return per trade.
        trade_count: Total number of trades.
        total_fees: Total trading fees paid.
        final_equity: Ending account balance.
        calmar_ratio: Return / max drawdown.
        avg_holding_period: Average time in position.
    """

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: timedelta = field(default_factory=timedelta)
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    trade_count: int = 0
    total_fees: float = 0.0
    final_equity: float = 0.0
    calmar_ratio: float = 0.0
    avg_holding_period: timedelta = field(default_factory=timedelta)

    # Internal data
    _returns: list[float] = field(default_factory=list, repr=False)
    _equity_curve: list[tuple[datetime, float]] = field(default_factory=list, repr=False)

    @classmethod
    def from_result(
        cls,
        result: BacktestResult,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252,
    ) -> PerformanceMetrics:
        """Calculate metrics from a BacktestResult.

        Args:
            result: Backtest result to analyze.
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            periods_per_year: Trading periods per year (252 for daily).

        Returns:
            PerformanceMetrics with all calculations.
        """
        metrics = cls()

        if not result.equity_curve:
            return metrics

        # Convert equity curve
        equity_curve = [(ts, float(eq)) for ts, eq in result.equity_curve]
        metrics._equity_curve = equity_curve

        initial_equity = equity_curve[0][1]
        final_equity = equity_curve[-1][1]

        # Basic returns
        metrics.final_equity = final_equity
        metrics.total_return = (
            (final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0
        )

        # Calculate period returns for risk metrics
        returns = cls._calculate_returns(equity_curve)
        metrics._returns = returns

        # Duration for annualization
        if result.start_time and result.end_time:
            duration = result.end_time - result.start_time
            years = duration.total_seconds() / (365.25 * 24 * 3600)
            if years > 0:
                metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1
            # Estimate periods per year based on data frequency
            if len(equity_curve) > 1:
                avg_period = duration.total_seconds() / len(equity_curve)
                periods_per_year = int(365.25 * 24 * 3600 / avg_period) if avg_period > 0 else 252

        # Risk metrics
        if returns:
            metrics.sharpe_ratio = cls._calculate_sharpe(returns, risk_free_rate, periods_per_year)
            metrics.sortino_ratio = cls._calculate_sortino(
                returns, risk_free_rate, periods_per_year
            )

        # Drawdown analysis
        dd_stats = cls._calculate_drawdown(equity_curve)
        metrics.max_drawdown = dd_stats["max_drawdown"]
        metrics.max_drawdown_duration = dd_stats["max_duration"]

        # Calmar ratio (annualized return / max drawdown)
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

        # Trade statistics
        metrics.trade_count = result.trade_count
        metrics.total_fees = sum(float(t.fee) for t in result.trades)

        trade_pnls = cls._calculate_trade_pnls(result)
        if trade_pnls:
            metrics.win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)
            metrics.avg_trade_return = sum(trade_pnls) / len(trade_pnls)
            metrics.profit_factor = cls._calculate_profit_factor(trade_pnls)

        # Average holding period
        metrics.avg_holding_period = cls._calculate_avg_holding_period(result)

        return metrics

    @staticmethod
    def _calculate_returns(equity_curve: list[tuple[datetime, float]]) -> list[float]:
        """Calculate period-over-period returns."""
        if len(equity_curve) < 2:
            return []

        returns = []
        for i in range(1, len(equity_curve)):
            prev_eq = equity_curve[i - 1][1]
            curr_eq = equity_curve[i][1]
            if prev_eq > 0:
                returns.append((curr_eq - prev_eq) / prev_eq)
        return returns

    @staticmethod
    def _calculate_sharpe(
        returns: list[float],
        risk_free_rate: float,
        periods_per_year: int,
    ) -> float:
        """Calculate Sharpe ratio.

        Sharpe = (mean_return - risk_free) / std_dev * sqrt(periods)
        """
        if len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        rf_per_period = risk_free_rate / periods_per_year

        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0

        if std_dev == 0:
            return 0.0

        return (mean_return - rf_per_period) / std_dev * math.sqrt(periods_per_year)

    @staticmethod
    def _calculate_sortino(
        returns: list[float],
        risk_free_rate: float,
        periods_per_year: int,
    ) -> float:
        """Calculate Sortino ratio (downside deviation only).

        Sortino = (mean_return - risk_free) / downside_dev * sqrt(periods)
        """
        if len(returns) < 2:
            return 0.0

        mean_return = sum(returns) / len(returns)
        rf_per_period = risk_free_rate / periods_per_year

        # Downside deviation (only negative returns)
        downside_returns = [min(0, r - rf_per_period) for r in returns]
        downside_variance = sum(r**2 for r in downside_returns) / len(downside_returns)
        downside_dev = math.sqrt(downside_variance) if downside_variance > 0 else 0.0

        if downside_dev == 0:
            return 0.0

        return (mean_return - rf_per_period) / downside_dev * math.sqrt(periods_per_year)

    @staticmethod
    def _calculate_drawdown(equity_curve: list[tuple[datetime, float]]) -> dict:
        """Calculate maximum drawdown and duration."""
        if not equity_curve:
            return {"max_drawdown": 0.0, "max_duration": timedelta()}

        peak = equity_curve[0][1]
        peak_time = equity_curve[0][0]
        max_drawdown = 0.0
        max_duration = timedelta()

        current_drawdown_start: datetime | None = None

        for timestamp, equity in equity_curve:
            if equity > peak:
                # New peak
                if current_drawdown_start is not None:
                    duration = timestamp - current_drawdown_start
                    if duration > max_duration:
                        max_duration = duration
                peak = equity
                peak_time = timestamp
                current_drawdown_start = None
            else:
                # In drawdown
                if current_drawdown_start is None:
                    current_drawdown_start = peak_time
                drawdown = (peak - equity) / peak if peak > 0 else 0.0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

        # Check if still in drawdown at end
        if current_drawdown_start is not None:
            duration = equity_curve[-1][0] - current_drawdown_start
            if duration > max_duration:
                max_duration = duration

        return {"max_drawdown": max_drawdown, "max_duration": max_duration}

    @staticmethod
    def _calculate_trade_pnls(result: BacktestResult) -> list[float]:
        """Calculate P&L for each round-trip trade.

        Pairs entry and exit trades to calculate individual trade returns.
        """
        trades = result.trades
        if len(trades) < 2:
            return []

        pnls = []
        i = 0
        while i < len(trades) - 1:
            entry = trades[i]
            exit_trade = trades[i + 1]

            # Simple pairing: entry followed by exit
            if entry.side == "buy" and exit_trade.side == "sell":
                pnl = float((exit_trade.price - entry.price) * entry.quantity)
                pnl -= float(entry.fee + exit_trade.fee)
                pnls.append(pnl)
                i += 2
            elif entry.side == "sell" and exit_trade.side == "buy":
                pnl = float((entry.price - exit_trade.price) * entry.quantity)
                pnl -= float(entry.fee + exit_trade.fee)
                pnls.append(pnl)
                i += 2
            else:
                i += 1

        return pnls

    @staticmethod
    def _calculate_profit_factor(pnls: list[float]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def _calculate_avg_holding_period(result: BacktestResult) -> timedelta:
        """Calculate average holding period across trades."""
        trades = result.trades
        if len(trades) < 2:
            return timedelta()

        holding_periods = []
        i = 0
        while i < len(trades) - 1:
            entry = trades[i]
            exit_trade = trades[i + 1]

            # Check for valid entry/exit pair
            if (entry.side == "buy" and exit_trade.side == "sell") or (
                entry.side == "sell" and exit_trade.side == "buy"
            ):
                period = exit_trade.timestamp - entry.timestamp
                holding_periods.append(period)
                i += 2
            else:
                i += 1

        if not holding_periods:
            return timedelta()

        total_seconds = sum(p.total_seconds() for p in holding_periods)
        avg_seconds = total_seconds / len(holding_periods)
        return timedelta(seconds=avg_seconds)

    def summary(self) -> str:
        """Generate a human-readable summary.

        Returns:
            Formatted string with key metrics.
        """
        lines = [
            "=" * 50,
            "BACKTEST PERFORMANCE SUMMARY",
            "=" * 50,
            f"Total Return:        {self.total_return:>10.2%}",
            f"Annualized Return:   {self.annualized_return:>10.2%}",
            f"Sharpe Ratio:        {self.sharpe_ratio:>10.2f}",
            f"Sortino Ratio:       {self.sortino_ratio:>10.2f}",
            f"Max Drawdown:        {self.max_drawdown:>10.2%}",
            f"Calmar Ratio:        {self.calmar_ratio:>10.2f}",
            "-" * 50,
            f"Trade Count:         {self.trade_count:>10}",
            f"Win Rate:            {self.win_rate:>10.2%}",
            f"Profit Factor:       {self.profit_factor:>10.2f}",
            f"Avg Trade Return:    {self.avg_trade_return:>10.2%}",
            f"Total Fees:          ${self.total_fees:>9.2f}",
            "-" * 50,
            f"Final Equity:        ${self.final_equity:>9.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)
