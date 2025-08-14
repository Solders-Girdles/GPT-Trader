"""Integration Orchestrator for GPT-Trader.

This module provides the main orchestrator that connects all components into one
working system for end-to-end backtesting and trading operations.

The orchestrator coordinates:
- Data pipeline (INT-002)
- Strategy-allocator bridge (INT-001)
- Risk management integration (INT-003)
- Execution and portfolio tracking
- Performance metrics and reporting
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from bot.config import get_config
from bot.dataflow.pipeline import DataPipeline, PipelineConfig
from bot.exec.ledger import Ledger
from bot.integration.strategy_allocator_bridge import StrategyAllocatorBridge
from bot.logging import get_logger
from bot.metrics.report import perf_metrics
from bot.portfolio.allocator import PortfolioRules
from bot.risk.integration import RiskIntegration, RiskConfig, AllocationResult
from bot.strategy.base import Strategy

logger = get_logger("orchestrator")


@dataclass
class BacktestConfig:
    """Configuration for integrated backtest."""

    # Date range
    start_date: datetime
    end_date: datetime

    # Capital
    initial_capital: float = 1_000_000.0

    # Data settings
    use_cache: bool = True
    strict_validation: bool = True

    # Risk settings
    risk_config: Optional[RiskConfig] = None

    # Portfolio settings
    portfolio_rules: Optional[PortfolioRules] = None

    # Output settings
    output_dir: str = "data/backtests"
    save_trades: bool = True
    save_portfolio: bool = True
    save_metrics: bool = True
    generate_plot: bool = True

    # Progress tracking
    show_progress: bool = True
    quiet_mode: bool = False

    def __post_init__(self):
        """Initialize defaults from config."""
        if self.risk_config is None:
            self.risk_config = RiskConfig()

        if self.portfolio_rules is None:
            config = get_config()
            self.portfolio_rules = PortfolioRules(
                per_trade_risk_pct=0.01,  # 1% risk per trade
                max_positions=10,
                max_gross_exposure_pct=0.95,  # 95% max exposure
                atr_k=2.0,
                cost_bps=5.0,  # 5 bps transaction costs
            )


@dataclass
class BacktestResults:
    """Results from integrated backtest run."""

    # Performance metrics
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Trading statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Risk metrics
    max_positions: int = 0
    avg_positions: float = 0.0
    total_costs: float = 0.0

    # Data
    equity_curve: Optional[pd.Series] = None
    trades: Optional[pd.DataFrame] = None
    positions: Optional[pd.DataFrame] = None
    risk_metrics: Dict[str, float] = field(default_factory=dict)

    # Execution info
    symbols_traded: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_days: int = 0
    execution_time_seconds: float = 0.0

    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            "performance": {
                "total_return": self.total_return,
                "cagr": self.cagr,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "volatility": self.volatility,
                "calmar_ratio": self.calmar_ratio,
                "sortino_ratio": self.sortino_ratio,
            },
            "trading": {
                "total_trades": self.total_trades,
                "winning_trades": self.winning_trades,
                "losing_trades": self.losing_trades,
                "win_rate": self.win_rate,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "profit_factor": self.profit_factor,
            },
            "risk": {
                "max_positions": self.max_positions,
                "avg_positions": self.avg_positions,
                "total_costs": self.total_costs,
                **self.risk_metrics,
            },
            "execution": {
                "symbols_traded": self.symbols_traded,
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "duration_days": self.duration_days,
                "execution_time_seconds": self.execution_time_seconds,
            },
            "issues": {
                "warnings": self.warnings,
                "errors": self.errors,
            },
        }


class IntegratedOrchestrator:
    """Main orchestrator connecting all GPT-Trader components.

    This class provides a unified interface for running complete backtests
    that integrate data pipeline, strategy execution, risk management,
    portfolio allocation, and performance tracking.

    Flow:
    Data Pipeline → Strategy Signals → Risk Validation → Allocation → Execution → Results
    """

    def __init__(self, config: BacktestConfig):
        """Initialize the orchestrator with configuration.

        Args:
            config: Backtest configuration parameters
        """
        self.config = config

        # Initialize components
        self.data_pipeline = DataPipeline(
            PipelineConfig(
                use_cache=config.use_cache,
                strict_validation=config.strict_validation,
                timeout_seconds=30.0,
                retry_attempts=2,
            )
        )

        self.risk_integration = RiskIntegration(
            risk_config=config.risk_config, portfolio_rules=config.portfolio_rules
        )

        self.ledger = Ledger()

        # State tracking
        self.current_equity = config.initial_capital
        self.current_positions: Dict[str, int] = {}
        self.current_prices: Dict[str, float] = {}
        self.daily_pnl = 0.0

        # Results accumulation
        self.equity_history: List[Tuple[datetime, float]] = []
        self.position_history: List[Dict[str, Any]] = []

        logger.info(
            f"IntegratedOrchestrator initialized with ${config.initial_capital:,.0f} "
            f"from {config.start_date.date()} to {config.end_date.date()}"
        )

    def run_backtest(self, strategy: Strategy, symbols: List[str], **kwargs) -> BacktestResults:
        """Run a complete integrated backtest.

        Args:
            strategy: Trading strategy to test
            symbols: List of symbols to trade
            **kwargs: Additional arguments passed to components

        Returns:
            BacktestResults with comprehensive performance data
        """
        start_time = time.time()

        if not self.config.quiet_mode:
            logger.info(f"Starting integrated backtest: {strategy.name} on {len(symbols)} symbols")

        # Initialize results
        results = BacktestResults(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            symbols_traded=symbols.copy(),
        )

        try:
            # Step 1: Load and validate data
            market_data = self._load_market_data(symbols, results)
            if not market_data:
                results.errors.append("No market data could be loaded")
                return results

            # Step 2: Create strategy-allocator bridge
            bridge = StrategyAllocatorBridge(strategy, self.config.portfolio_rules)
            if not bridge.validate_configuration():
                results.errors.append("Strategy-allocator bridge validation failed")
                return results

            # Step 3: Get trading dates
            trading_dates = self._get_trading_dates(market_data)
            results.duration_days = len(trading_dates)

            if not self.config.quiet_mode:
                logger.info(
                    f"Running backtest over {len(trading_dates)} trading days "
                    f"with {len(market_data)} symbols"
                )

            # Step 4: Daily trading loop
            self._run_daily_trading_loop(bridge, market_data, trading_dates, results)

            # Step 5: Calculate final performance metrics
            self._calculate_performance_metrics(results)

            # Step 6: Generate outputs
            self._generate_outputs(strategy, results)

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            results.errors.append(f"Backtest execution failed: {str(e)}")

        results.execution_time_seconds = time.time() - start_time

        if not self.config.quiet_mode:
            self._log_final_results(results)

        return results

    def _load_market_data(
        self, symbols: List[str], results: BacktestResults
    ) -> Dict[str, pd.DataFrame]:
        """Load and validate market data for all symbols."""
        try:
            if not self.config.quiet_mode:
                logger.info(f"Loading market data for {len(symbols)} symbols...")

            market_data = self.data_pipeline.fetch_and_validate(
                symbols=symbols,
                start=self.config.start_date - timedelta(days=365),  # Extra history for indicators
                end=self.config.end_date,
                use_cache=self.config.use_cache,
            )

            # Log data quality metrics
            metrics = self.data_pipeline.get_metrics()
            if metrics.symbols_failed > 0:
                results.warnings.append(
                    f"Failed to load {metrics.symbols_failed} symbols: {metrics.errors[:3]}"
                )

            if not self.config.quiet_mode:
                logger.info(
                    f"Loaded data for {len(market_data)}/{len(symbols)} symbols "
                    f"({metrics.success_rate:.1f}% success rate)"
                )

            return market_data

        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            results.errors.append(f"Data loading failed: {str(e)}")
            return {}

    def _get_trading_dates(self, market_data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Get sorted list of trading dates from market data."""
        all_dates = set()

        for symbol, df in market_data.items():
            # Filter to backtest date range
            df_filtered = df[
                (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
            ]
            all_dates.update(df_filtered.index.to_pydatetime())

        trading_dates = sorted(list(all_dates))
        logger.debug(f"Found {len(trading_dates)} unique trading dates")

        return trading_dates

    def _run_daily_trading_loop(
        self,
        bridge: StrategyAllocatorBridge,
        market_data: Dict[str, pd.DataFrame],
        trading_dates: List[datetime],
        results: BacktestResults,
    ) -> None:
        """Execute the main daily trading loop."""

        # Initialize progress tracking
        if self.config.show_progress:
            try:
                from tqdm import tqdm

                date_iter = tqdm(trading_dates, desc="Backtest Progress", leave=False)
            except ImportError:
                date_iter = trading_dates
        else:
            date_iter = trading_dates

        position_counts = []

        for current_date in date_iter:
            try:
                # Get today's market data snapshot
                daily_data = self._get_daily_data(market_data, current_date)

                # Update current prices
                self._update_current_prices(daily_data)

                # Calculate overnight P&L
                overnight_pnl = self._calculate_overnight_pnl()
                self.current_equity += overnight_pnl
                self.daily_pnl = overnight_pnl

                # Generate signals and allocations using bridge
                allocations = bridge.process_signals(daily_data, self.current_equity)

                # Apply risk management
                risk_result = self.risk_integration.validate_allocations(
                    allocations=allocations,
                    current_prices=self.current_prices,
                    portfolio_value=self.current_equity,
                    market_data=daily_data,
                    current_positions=self.current_positions,
                )

                # Execute trades based on risk-adjusted allocations
                trades_executed = self._execute_trades(
                    risk_result.adjusted_allocations, current_date
                )

                # Calculate intraday P&L
                intraday_pnl = self._calculate_intraday_pnl(daily_data)
                self.current_equity += intraday_pnl
                self.daily_pnl += intraday_pnl

                # Record daily state
                self.equity_history.append((current_date, self.current_equity))

                num_positions = len([q for q in self.current_positions.values() if q > 0])
                position_counts.append(num_positions)

                self.position_history.append(
                    {
                        "date": current_date,
                        "equity": self.current_equity,
                        "positions": num_positions,
                        "daily_pnl": self.daily_pnl,
                        "trades_executed": trades_executed,
                        "risk_warnings": len(risk_result.warnings),
                        "total_exposure": risk_result.total_exposure,
                    }
                )

                # Check daily loss limits
                if self.risk_integration.check_daily_loss_limit(self.daily_pnl):
                    results.warnings.append(
                        f"Daily loss limit breached on {current_date.date()}: {self.daily_pnl:.2f}"
                    )

            except Exception as e:
                logger.warning(f"Error processing {current_date.date()}: {e}")
                results.warnings.append(f"Trading error on {current_date.date()}: {str(e)}")

        # Store position statistics
        if position_counts:
            results.max_positions = max(position_counts)
            results.avg_positions = np.mean(position_counts)

    def _get_daily_data(
        self, market_data: Dict[str, pd.DataFrame], current_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Get market data snapshot up to current date for signal generation."""
        daily_data = {}

        for symbol, df in market_data.items():
            # Get data up to (and including) current date
            mask = df.index <= current_date
            symbol_data = df[mask].copy()

            if not symbol_data.empty:
                daily_data[symbol] = symbol_data

        return daily_data

    def _update_current_prices(self, daily_data: Dict[str, pd.DataFrame]) -> None:
        """Update current prices from latest market data."""
        for symbol, df in daily_data.items():
            if not df.empty and "Close" in df.columns:
                self.current_prices[symbol] = float(df["Close"].iloc[-1])

    def _calculate_overnight_pnl(self) -> float:
        """Calculate P&L from price changes since last update."""
        # For simplicity, assuming we calculate P&L from current positions
        # In a real implementation, this would track overnight gaps
        return 0.0  # Placeholder - would need previous close prices

    def _calculate_intraday_pnl(self, daily_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate intraday P&L for current positions."""
        intraday_pnl = 0.0

        for symbol, quantity in self.current_positions.items():
            if quantity == 0 or symbol not in daily_data:
                continue

            df = daily_data[symbol]
            if df.empty or "Open" not in df.columns or "Close" not in df.columns:
                continue

            # Intraday return: (Close - Open) / Open
            open_price = float(df["Open"].iloc[-1])
            close_price = float(df["Close"].iloc[-1])

            if open_price > 0:
                price_change = close_price - open_price
                position_pnl = quantity * price_change
                intraday_pnl += position_pnl

        return intraday_pnl

    def _execute_trades(self, target_allocations: Dict[str, int], trade_date: datetime) -> int:
        """Execute trades to reach target allocations."""
        trades_executed = 0

        # Calculate trades needed
        all_symbols = set(self.current_positions.keys()) | set(target_allocations.keys())

        for symbol in all_symbols:
            current_qty = self.current_positions.get(symbol, 0)
            target_qty = target_allocations.get(symbol, 0)

            if current_qty != target_qty:
                # Execute trade
                price = self.current_prices.get(symbol, 0.0)
                if price > 0:
                    self.ledger.submit_and_fill(
                        symbol=symbol,
                        new_qty=target_qty,
                        price=price,
                        ts=trade_date,
                        reason="rebalance",
                        cost_usd=abs(target_qty - current_qty) * price * 0.0005,  # 5 bps cost
                    )
                    trades_executed += 1

                # Update positions
                if target_qty == 0:
                    self.current_positions.pop(symbol, None)
                else:
                    self.current_positions[symbol] = target_qty

        return trades_executed

    def _calculate_performance_metrics(self, results: BacktestResults) -> None:
        """Calculate comprehensive performance metrics."""
        if not self.equity_history:
            return

        # Create equity curve
        dates, equity_values = zip(*self.equity_history)
        equity_series = pd.Series(equity_values, index=pd.DatetimeIndex(dates))
        results.equity_curve = equity_series

        # Calculate standard performance metrics
        if len(equity_series) > 1:
            perf = perf_metrics(equity_series)
            results.total_return = perf["total_return"]
            results.cagr = perf["cagr"]
            results.sharpe_ratio = perf["sharpe"]
            results.max_drawdown = perf["max_drawdown"]
            results.volatility = perf["vol"]

        # Calculate trading statistics
        trades_df = self.ledger.to_trades_dataframe()
        results.trades = trades_df

        if not trades_df.empty:
            results.total_trades = len(trades_df)

            # Calculate win/loss statistics
            pnl_trades = trades_df[trades_df["pnl"].notna()]
            if not pnl_trades.empty:
                winning_trades = pnl_trades[pnl_trades["pnl"] > 0]
                losing_trades = pnl_trades[pnl_trades["pnl"] < 0]

                results.winning_trades = len(winning_trades)
                results.losing_trades = len(losing_trades)

                if results.total_trades > 0:
                    results.win_rate = results.winning_trades / results.total_trades

                if len(winning_trades) > 0:
                    results.avg_win = winning_trades["pnl"].mean()

                if len(losing_trades) > 0:
                    results.avg_loss = abs(losing_trades["pnl"].mean())

                if results.avg_loss > 0:
                    results.profit_factor = results.avg_win / results.avg_loss

        # Calculate additional metrics
        if equity_series.std() != 0:
            returns = equity_series.pct_change().dropna()
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std() * np.sqrt(252)
                results.sortino_ratio = (
                    (returns.mean() * 252) / downside_std if downside_std != 0 else 0.0
                )

            if results.max_drawdown != 0:
                results.calmar_ratio = results.cagr / results.max_drawdown

        # Add risk metrics from risk integration
        risk_report = self.risk_integration.generate_risk_report()
        results.risk_metrics = {
            "final_portfolio_value": risk_report["portfolio_value"],
            "final_daily_pnl": risk_report["daily_pnl"],
        }

    def _generate_outputs(self, strategy: Strategy, results: BacktestResults) -> None:
        """Generate output files and plots."""
        if not any(
            [
                self.config.save_trades,
                self.config.save_portfolio,
                self.config.save_metrics,
                self.config.generate_plot,
            ]
        ):
            return

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{strategy.name}_integrated_{timestamp}"

        # Save equity curve
        if self.config.save_portfolio and results.equity_curve is not None:
            portfolio_file = output_dir / f"{base_name}_portfolio.csv"
            results.equity_curve.to_csv(portfolio_file)
            logger.info(f"Portfolio saved to {portfolio_file}")

        # Save trades
        if self.config.save_trades and results.trades is not None:
            trades_file = output_dir / f"{base_name}_trades.csv"
            results.trades.to_csv(trades_file, index=False)
            logger.info(f"Trades saved to {trades_file}")

        # Save metrics summary
        if self.config.save_metrics:
            metrics_file = output_dir / f"{base_name}_metrics.csv"
            metrics_dict = results.to_dict()

            # Flatten nested dict for CSV
            flattened = {}
            for category, values in metrics_dict.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        flattened[f"{category}_{key}"] = value
                else:
                    flattened[category] = values

            pd.Series(flattened).to_csv(metrics_file)
            logger.info(f"Metrics saved to {metrics_file}")

        # Generate plot
        if self.config.generate_plot and results.equity_curve is not None:
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 6))
                plt.plot(results.equity_curve.index, results.equity_curve.values)
                plt.title(f"Equity Curve - {strategy.name} (Integrated Backtest)")
                plt.xlabel("Date")
                plt.ylabel("Portfolio Value ($)")
                plt.grid(True, alpha=0.3)

                plot_file = output_dir / f"{base_name}_equity_curve.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close()

                logger.info(f"Equity curve plot saved to {plot_file}")

            except Exception as e:
                logger.warning(f"Failed to generate plot: {e}")
                results.warnings.append(f"Plot generation failed: {str(e)}")

    def _log_final_results(self, results: BacktestResults) -> None:
        """Log final backtest results summary."""
        logger.info("=== INTEGRATED BACKTEST RESULTS ===")
        logger.info(f"Total Return: {results.total_return:.2%}")
        logger.info(f"CAGR: {results.cagr:.2%}")
        logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
        logger.info(f"Volatility: {results.volatility:.2%}")
        logger.info(f"Total Trades: {results.total_trades}")
        logger.info(f"Win Rate: {results.win_rate:.2%}")
        logger.info(f"Max Positions: {results.max_positions}")
        logger.info(f"Avg Positions: {results.avg_positions:.1f}")
        logger.info(f"Duration: {results.duration_days} days")
        logger.info(f"Execution Time: {results.execution_time_seconds:.2f}s")

        if results.warnings:
            logger.warning(f"Warnings: {len(results.warnings)}")

        if results.errors:
            logger.error(f"Errors: {len(results.errors)}")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health = {"status": "healthy", "components": {}, "warnings": [], "errors": []}

        # Check data pipeline
        try:
            pipeline_health = self.data_pipeline.health_check()
            health["components"]["data_pipeline"] = pipeline_health
            if pipeline_health["status"] != "healthy":
                health["status"] = "degraded"
                health["warnings"].extend(pipeline_health.get("warnings", []))
                health["errors"].extend(pipeline_health.get("errors", []))
        except Exception as e:
            health["status"] = "unhealthy"
            health["errors"].append(f"Data pipeline health check failed: {str(e)}")

        # Check risk integration
        try:
            risk_report = self.risk_integration.generate_risk_report()
            health["components"]["risk_integration"] = {
                "status": "healthy",
                "portfolio_value": risk_report["portfolio_value"],
            }
        except Exception as e:
            health["status"] = "degraded"
            health["warnings"].append(f"Risk integration check failed: {str(e)}")

        return health


# Convenience function for simple backtest execution
def run_integrated_backtest(
    strategy: Strategy,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 1_000_000.0,
    **kwargs,
) -> BacktestResults:
    """Run an integrated backtest with default configuration.

    Args:
        strategy: Trading strategy to test
        symbols: List of symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital amount
        **kwargs: Additional configuration options

    Returns:
        BacktestResults with comprehensive performance data
    """
    config = BacktestConfig(
        start_date=start_date, end_date=end_date, initial_capital=initial_capital, **kwargs
    )

    orchestrator = IntegratedOrchestrator(config)
    return orchestrator.run_backtest(strategy, symbols)
