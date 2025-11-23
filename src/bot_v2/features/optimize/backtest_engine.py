"""
Production-parity backtesting engine.

This engine reuses the production BaselinePerpsStrategy.decide() loop,
ensuring perfect alignment between backtest and live execution decisions.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from bot_v2.utilities.logging_patterns import get_logger

from .backtest_portfolio import BacktestPortfolio
from .backtester import calculate_metrics
from .decision_logger import DecisionLogger
from .types_v2 import BacktestConfig, BacktestResult, DecisionContext

logger = get_logger(__name__, component="optimize")


class BacktestEngine:
    """
    Backtesting engine that reuses production strategy code.

    This ensures that backtests use the exact same decision logic as live trading,
    eliminating code duplication and guaranteeing parity.
    """

    def __init__(
        self,
        *,
        strategy: BaselinePerpsStrategy,
        config: BacktestConfig | None = None,
    ):
        """
        Initialize backtest engine.

        Args:
            strategy: Production strategy instance to test
            config: Backtest configuration
        """
        self.strategy = strategy
        self.config = config or BacktestConfig()

        self.portfolio = BacktestPortfolio(
            initial_capital=self.config.initial_capital,
            commission_rate=self.config.commission_rate,
            slippage_rate=self.config.slippage_rate,
        )

        self.decision_logger = DecisionLogger(
            enabled=self.config.enable_decision_logging,
            base_directory=self.config.log_directory,
        )

    def run(
        self,
        *,
        data: pd.DataFrame,
        symbol: str,
        product: Product | None = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            data: DataFrame with columns: timestamp, open, high, low, close, volume
            symbol: Trading symbol
            product: Product metadata (will create default if None)

        Returns:
            BacktestResult with decisions, metrics, and equity curve
        """
        # Validate data
        required_cols = ["close"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        if len(data) == 0:
            raise ValueError("Data is empty")

        # Create default product if not provided
        if product is None:
            product = self._create_default_product(symbol)

        # Get timestamps
        if "timestamp" in data.columns:
            timestamps = [self._to_datetime(ts) for ts in data["timestamp"]]
        else:
            timestamps = [self._to_datetime(idx) for idx in data.index]

        start_time = timestamps[0]
        end_time = timestamps[-1]

        # Generate run ID
        run_id = f"bt_{start_time.strftime('%Y%m%d_%H%M%S')}_{symbol}"

        logger.info(
            "Starting backtest | run_id=%s | symbol=%s | bars=%d | period=%s to %s",
            run_id,
            symbol,
            len(data),
            start_time.date(),
            end_time.date(),
        )

        # Clear any previous strategy state
        self.strategy.reset(symbol)
        self.decision_logger.clear()

        # Run simulation
        self._simulate(data=data, symbol=symbol, product=product, timestamps=timestamps)

        # Calculate metrics
        equity_curve_series = pd.Series(
            [float(eq) for _, eq in self.portfolio.equity_history],
            index=[ts for ts, _ in self.portfolio.equity_history],
        )

        # Build trade list for metrics calculation
        trades = self._build_trade_list()
        metrics = calculate_metrics(trades, equity_curve_series)

        # Build result
        result = BacktestResult(
            run_id=run_id,
            strategy_name=self.strategy.__class__.__name__,
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            config=self.config,
            decisions=self.decision_logger.decisions,
            metrics=metrics,
            equity_curve=self.portfolio.equity_history,
        )

        # Save decision log if enabled
        if self.config.enable_decision_logging:
            log_path = self.decision_logger.save(result)
            logger.info("Decision log saved | path=%s", log_path)

        logger.info(
            "Backtest complete | run_id=%s | total_return=%.2f%% | sharpe=%.2f | max_dd=%.2f%% | trades=%d",
            run_id,
            metrics.total_return * 100,
            metrics.sharpe_ratio,
            metrics.max_drawdown * 100,
            metrics.total_trades,
        )

        return result

    def _simulate(
        self,
        *,
        data: pd.DataFrame,
        symbol: str,
        product: Product,
        timestamps: list[datetime],
    ) -> None:
        """Run the simulation loop."""
        # Get MA periods from strategy config
        long_period = self.strategy.config.long_ma_period

        # Need enough bars for MA calculation
        if len(data) < long_period:
            logger.warning(
                "Insufficient data for MA calculation | need=%d | have=%d",
                long_period,
                len(data),
            )
            return

        # Extract close prices
        closes = [Decimal(str(price)) for price in data["close"].values]

        # Iterate through each bar (starting after we have enough history for MA)
        for i in range(long_period, len(data)):
            timestamp = timestamps[i]
            current_mark = closes[i]

            # Build recent_marks history (excluding current)
            recent_marks = closes[max(0, i - long_period - 10) : i]

            # Get current position state
            position_state = self.portfolio.get_position_state(symbol)

            # Get current equity
            equity = self.portfolio.get_equity({symbol: current_mark})

            # Call production strategy.decide() - THIS IS THE KEY INTEGRATION POINT
            decision = self.strategy.decide(
                symbol=symbol,
                current_mark=current_mark,
                position_state=position_state,
                recent_marks=recent_marks,
                equity=equity,
                product=product,
            )

            # Extract signal metadata if available
            signal_label = None
            signal_metadata = None
            if hasattr(self.strategy, "state") and hasattr(self.strategy.state, "last_signal"):
                # This would require enhancing the strategy to expose signal data
                pass

            # Create decision context
            context = DecisionContext(
                timestamp=timestamp,
                symbol=symbol,
                current_mark=current_mark,
                recent_marks=list(recent_marks),
                position_state=position_state,
                equity=equity,
                signal_label=signal_label,
                signal_metadata=signal_metadata,
            )

            # Execute decision
            execution = self.portfolio.process_decision(
                decision=decision,
                symbol=symbol,
                current_price=current_mark,
                product=product,
                timestamp=timestamp,
            )

            # Log decision
            self.decision_logger.log_decision(
                context=context,
                decision=decision,
                execution=execution,
            )

            # Record equity snapshot
            self.portfolio.record_equity(timestamp, {symbol: current_mark})

    def _build_trade_list(self) -> list[dict[str, Any]]:
        """
        Build trade list from decision log for metrics calculation.

        This converts our DecisionRecords into the format expected by
        the existing calculate_metrics() function.
        """
        trades: list[dict[str, Any]] = []
        entry_price = None

        for record in self.decision_logger.decisions:
            if not record.execution.filled:
                continue

            exec_result = record.execution
            decision = record.decision

            if decision.action.value in ["buy", "sell"]:
                # Entry trade
                trades.append(
                    {
                        "date": record.context.timestamp,
                        "type": decision.action.value,
                        "price": float(exec_result.fill_price or 0),
                        "shares": float(exec_result.filled_quantity or 0),
                        "value": float(
                            (exec_result.fill_price or 0) * (exec_result.filled_quantity or 0)
                        ),
                    }
                )
                entry_price = exec_result.fill_price

            elif decision.action.value == "close":
                # Exit trade
                if entry_price is not None and exec_result.fill_price is not None:
                    exit_price = exec_result.fill_price
                    quantity = exec_result.filled_quantity or Decimal("0")

                    # Calculate P&L (simplified - assumes long positions)
                    # TODO: Handle short positions correctly
                    pnl = float((exit_price - entry_price) * quantity)
                    pnl_pct = float((exit_price - entry_price) / entry_price) if entry_price else 0

                    trades.append(
                        {
                            "date": record.context.timestamp,
                            "type": "sell",  # Use "sell" for compatibility with calculate_metrics
                            "price": float(exit_price),
                            "shares": float(quantity),
                            "value": float(exit_price * quantity),
                            "pnl": pnl,
                            "pnl_pct": pnl_pct,
                        }
                    )
                    entry_price = None

        return trades

    def _create_default_product(self, symbol: str) -> Product:
        """Create a default product for backtesting."""
        return Product(
            symbol=symbol,
            base_asset=symbol.split("-")[0] if "-" in symbol else symbol,
            quote_asset="USD",
            market_type=MarketType.SPOT,
            min_size=Decimal("0.0001"),
            step_size=Decimal("0.0001"),
            min_notional=Decimal("1"),
            price_increment=Decimal("0.01"),
        )

    def _to_datetime(self, value: Any) -> datetime:
        """Convert various timestamp formats to datetime."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, (int, float)):
            # Assume Unix timestamp
            return datetime.fromtimestamp(value, tz=timezone.utc)
        return value


def run_backtest_production(
    *,
    strategy: BaselinePerpsStrategy,
    data: pd.DataFrame,
    symbol: str,
    product: Product | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Convenience function to run a production-parity backtest.

    Args:
        strategy: Production strategy instance
        data: Historical OHLC data
        symbol: Trading symbol
        product: Product metadata (optional)
        config: Backtest configuration (optional)

    Returns:
        BacktestResult with full decision log and metrics
    """
    engine = BacktestEngine(strategy=strategy, config=config)
    return engine.run(data=data, symbol=symbol, product=product)


__all__ = ["BacktestEngine", "run_backtest_production"]
