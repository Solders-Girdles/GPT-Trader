"""Backtest report generation."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from gpt_trader.backtesting.types import BacktestResult

from .risk import RiskMetrics, calculate_risk_metrics
from .statistics import TradeStatistics, calculate_trade_statistics

if TYPE_CHECKING:
    from gpt_trader.backtesting.simulation.broker import SimulatedBroker


class BacktestReporter:
    """
    Generates comprehensive backtest reports.

    Aggregates trade statistics, risk metrics, and simulation metadata
    into a complete BacktestResult.
    """

    def __init__(self, broker: SimulatedBroker):
        """
        Initialize reporter with a completed backtest.

        Args:
            broker: SimulatedBroker after backtest completion
        """
        self.broker = broker
        self._trade_stats: TradeStatistics | None = None
        self._risk_metrics: RiskMetrics | None = None

    @property
    def trade_statistics(self) -> TradeStatistics:
        """Get or compute trade statistics."""
        if self._trade_stats is None:
            self._trade_stats = calculate_trade_statistics(self.broker)
        return self._trade_stats

    @property
    def risk_metrics(self) -> RiskMetrics:
        """Get or compute risk metrics."""
        if self._risk_metrics is None:
            self._risk_metrics = calculate_risk_metrics(self.broker)
        return self._risk_metrics

    def generate_result(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResult:
        """
        Generate a complete BacktestResult.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            BacktestResult with all metrics populated
        """
        stats = self.trade_statistics
        risk = self.risk_metrics
        broker_stats = self.broker.get_statistics()

        duration_days = (end_date - start_date).days

        return BacktestResult(
            # Time range
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            # Performance
            initial_equity=broker_stats["initial_equity"],
            final_equity=broker_stats["final_equity"],
            total_return=broker_stats["total_return_pct"],
            total_return_usd=broker_stats["total_return_usd"],
            # PnL breakdown
            realized_pnl=stats.gross_profit + stats.gross_loss,
            unrealized_pnl=sum(
                (p.unrealized_pnl for p in self.broker.positions.values()),
                Decimal("0"),
            ),
            funding_pnl=broker_stats["funding_pnl"],
            fees_paid=broker_stats["total_fees_paid"],
            # Trade statistics
            total_trades=stats.total_trades,
            winning_trades=stats.winning_trades,
            losing_trades=stats.losing_trades,
            win_rate=stats.win_rate,
            # Risk metrics
            max_drawdown=risk.max_drawdown_pct,
            max_drawdown_usd=risk.max_drawdown_usd,
            sharpe_ratio=risk.sharpe_ratio,
            sortino_ratio=risk.sortino_ratio,
            # Position statistics
            avg_position_size_usd=stats.avg_position_size_usd,
            max_position_size_usd=stats.max_position_size_usd,
            avg_leverage=stats.avg_leverage,
            max_leverage=stats.max_leverage,
            # Execution quality
            avg_slippage_bps=stats.avg_slippage_bps,
            limit_fill_rate=stats.limit_fill_rate,
        )

    def generate_summary(self) -> str:
        """
        Generate a human-readable summary of the backtest.

        Returns:
            Formatted string summary
        """
        stats = self.trade_statistics
        risk = self.risk_metrics
        broker_stats = self.broker.get_statistics()

        lines = [
            "=" * 60,
            "BACKTEST RESULTS SUMMARY",
            "=" * 60,
            "",
            "PERFORMANCE",
            "-" * 40,
            f"  Initial Equity:     ${broker_stats['initial_equity']:,.2f}",
            f"  Final Equity:       ${broker_stats['final_equity']:,.2f}",
            f"  Total Return:       {broker_stats['total_return_pct']:.2f}%",
            f"  Total Return (USD): ${broker_stats['total_return_usd']:,.2f}",
            "",
            "RISK METRICS",
            "-" * 40,
            f"  Max Drawdown:       {risk.max_drawdown_pct:.2f}%",
            f"  Max Drawdown (USD): ${risk.max_drawdown_usd:,.2f}",
            (
                f"  Sharpe Ratio:       {risk.sharpe_ratio:.2f}"
                if risk.sharpe_ratio
                else "  Sharpe Ratio:       N/A"
            ),
            (
                f"  Sortino Ratio:      {risk.sortino_ratio:.2f}"
                if risk.sortino_ratio
                else "  Sortino Ratio:      N/A"
            ),
            f"  Volatility (Ann):   {risk.volatility_annualized:.2f}%",
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"  Total Trades:       {stats.total_trades}",
            f"  Winning Trades:     {stats.winning_trades}",
            f"  Losing Trades:      {stats.losing_trades}",
            f"  Win Rate:           {stats.win_rate:.2f}%",
            f"  Profit Factor:      {stats.profit_factor:.2f}",
            "",
            "COSTS",
            "-" * 40,
            f"  Total Fees:         ${stats.total_fees_paid:,.2f}",
            f"  Avg Slippage:       {stats.avg_slippage_bps:.2f} bps",
            f"  Funding PnL:        ${broker_stats['funding_pnl']:,.2f}",
            "",
            "=" * 60,
        ]

        return "\n".join(lines)

    def generate_csv_row(self) -> dict[str, str | int | float]:
        """
        Generate a dictionary suitable for CSV export.

        Returns:
            Dictionary with all metrics as string/numeric values
        """
        stats = self.trade_statistics
        risk = self.risk_metrics
        broker_stats = self.broker.get_statistics()

        return {
            "initial_equity": float(broker_stats["initial_equity"]),
            "final_equity": float(broker_stats["final_equity"]),
            "total_return_pct": float(broker_stats["total_return_pct"]),
            "total_return_usd": float(broker_stats["total_return_usd"]),
            "max_drawdown_pct": float(risk.max_drawdown_pct),
            "max_drawdown_usd": float(risk.max_drawdown_usd),
            "sharpe_ratio": float(risk.sharpe_ratio) if risk.sharpe_ratio else 0.0,
            "sortino_ratio": float(risk.sortino_ratio) if risk.sortino_ratio else 0.0,
            "total_trades": stats.total_trades,
            "winning_trades": stats.winning_trades,
            "losing_trades": stats.losing_trades,
            "win_rate_pct": float(stats.win_rate),
            "profit_factor": float(stats.profit_factor),
            "total_fees": float(stats.total_fees_paid),
            "avg_slippage_bps": float(stats.avg_slippage_bps),
            "funding_pnl": float(broker_stats["funding_pnl"]),
        }


def generate_backtest_report(
    broker: SimulatedBroker,
    start_date: datetime,
    end_date: datetime,
) -> BacktestResult:
    """
    Convenience function to generate a BacktestResult.

    Args:
        broker: Completed SimulatedBroker
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        BacktestResult with all metrics
    """
    reporter = BacktestReporter(broker)
    return reporter.generate_result(start_date, end_date)
