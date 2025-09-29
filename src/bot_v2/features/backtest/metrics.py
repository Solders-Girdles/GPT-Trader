"""
Performance metrics calculation for backtesting.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from ...config import get_config
from ...errors import BacktestError, ValidationError
from ...validation import PositiveNumberValidator, SeriesValidator, validate_inputs
from .types import BacktestMetrics, TradeDict

logger = logging.getLogger(__name__)

# Suppress specific warnings for cleaner output
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide"
)


@validate_inputs(
    equity_curve=SeriesValidator(),
    returns=SeriesValidator(),
    initial_capital=PositiveNumberValidator(),
)
def calculate_metrics(
    trades: list[TradeDict], equity_curve: pd.Series, returns: pd.Series, initial_capital: float
) -> BacktestMetrics:
    """
    Calculate performance metrics from backtest results.

    Args:
        trades: List of executed trades
        equity_curve: Portfolio equity over time
        returns: Daily returns
        initial_capital: Starting capital

    Returns:
        BacktestMetrics object

    Raises:
        BacktestError: If metric calculation fails
        ValidationError: If inputs are invalid
    """
    logger.debug(
        f"Calculating metrics for {len(trades)} trades and {len(equity_curve)} equity points",
        extra={
            "trades_count": len(trades),
            "equity_points": len(equity_curve),
            "returns_points": len(returns),
            "initial_capital": initial_capital,
        },
    )
    # Get configuration for metrics calculation
    config = get_config("backtest")
    perf_config = config.get("performance_metrics", {})

    # Validate inputs
    _validate_metrics_inputs(trades, equity_curve, returns, initial_capital)

    try:
        # Initialize metrics object
        metrics = BacktestMetrics()

        # Calculate basic return metrics with error handling
        metrics = _calculate_return_metrics(metrics, equity_curve, initial_capital, perf_config)

        # Calculate risk metrics with error handling
        metrics = _calculate_risk_metrics(metrics, returns, equity_curve, perf_config)

        # Calculate trade statistics with error handling
        metrics = _calculate_trade_statistics(metrics, trades)

        # Calculate advanced metrics if enabled
        if perf_config.get("calculate_sharpe", True):
            metrics = _calculate_sharpe_ratio(metrics, returns, perf_config)

        if perf_config.get("calculate_sortino", True):
            metrics = _calculate_sortino_ratio(metrics, returns, perf_config)

        if perf_config.get("calculate_calmar", True):
            metrics = _calculate_calmar_ratio(metrics, returns, equity_curve)

        logger.info(
            "Metrics calculated successfully",
            extra={
                "total_return": getattr(metrics, "total_return", 0),
                "sharpe_ratio": getattr(metrics, "sharpe_ratio", 0),
                "max_drawdown": getattr(metrics, "max_drawdown", 0),
                "total_trades": getattr(metrics, "total_trades", 0),
            },
        )

        return metrics

    except Exception as e:
        if isinstance(e, (BacktestError, ValidationError)):
            raise
        raise BacktestError(
            "Failed to calculate performance metrics",
            context={
                "trades_count": len(trades),
                "equity_points": len(equity_curve),
                "returns_points": len(returns),
                "initial_capital": initial_capital,
                "error": str(e),
            },
        ) from e


def _validate_metrics_inputs(
    trades: list[dict], equity_curve: pd.Series, returns: pd.Series, initial_capital: float
) -> None:
    """
    Validate inputs for metrics calculation.

    Args:
        trades: List of trades
        equity_curve: Portfolio equity over time
        returns: Daily returns
        initial_capital: Starting capital

    Raises:
        ValidationError: If validation fails
    """
    # Validate trades structure
    if not isinstance(trades, list):
        raise ValidationError("Trades must be a list", field="trades", value=type(trades).__name__)

    # Validate equity curve
    if equity_curve.empty:
        raise ValidationError("Equity curve cannot be empty", field="equity_curve")

    if equity_curve.isna().all():
        raise ValidationError("Equity curve contains only NaN values", field="equity_curve")

    # Validate returns
    if returns.empty:
        raise ValidationError("Returns series cannot be empty", field="returns")

    # Check alignment
    if len(equity_curve) != len(returns):
        raise ValidationError(
            "Equity curve and returns must have same length",
            field="series_alignment",
            value=f"equity: {len(equity_curve)}, returns: {len(returns)}",
        )


def _calculate_return_metrics(
    metrics: BacktestMetrics, equity_curve: pd.Series, initial_capital: float, config: dict
) -> BacktestMetrics:
    """Calculate return-based metrics."""
    try:
        if len(equity_curve) > 0:
            final_value = float(equity_curve.iloc[-1])

            # Handle edge cases
            if np.isnan(final_value) or np.isinf(final_value):
                logger.warning("Final equity value is invalid, using initial capital")
                final_value = initial_capital

            if initial_capital > 0:
                metrics.total_return = ((final_value - initial_capital) / initial_capital) * 100
            else:
                metrics.total_return = 0.0
                logger.warning("Initial capital is zero or negative")

        return metrics

    except Exception as e:
        logger.warning(f"Failed to calculate return metrics: {e}")
        metrics.total_return = 0.0
        return metrics


def _calculate_risk_metrics(
    metrics: BacktestMetrics, returns: pd.Series, equity_curve: pd.Series, config: dict
) -> BacktestMetrics:
    """Calculate risk-based metrics."""
    try:
        # Maximum drawdown
        if len(equity_curve) > 1:
            equity_clean = equity_curve.dropna()
            if len(equity_clean) > 1:
                rolling_max = equity_clean.expanding().max()
                drawdown = (equity_clean - rolling_max) / rolling_max

                # Handle edge cases
                drawdown = drawdown.replace([np.inf, -np.inf], 0)
                drawdown = drawdown.dropna()

                if len(drawdown) > 0:
                    metrics.max_drawdown = abs(float(drawdown.min())) * 100
                else:
                    metrics.max_drawdown = 0.0
            else:
                metrics.max_drawdown = 0.0
        else:
            metrics.max_drawdown = 0.0

        return metrics

    except Exception as e:
        logger.warning(f"Failed to calculate risk metrics: {e}")
        metrics.max_drawdown = 0.0
        return metrics


def _calculate_trade_statistics(metrics: BacktestMetrics, trades: list[dict]) -> BacktestMetrics:
    """Calculate trade-based statistics."""
    try:
        if not trades:
            metrics.total_trades = 0
            metrics.win_rate = 0.0
            metrics.profit_factor = 0.0
            return metrics

        # Count round-trip trades (buy-sell pairs)
        buy_trades = [t for t in trades if t.get("side") == "buy"]
        sell_trades = [t for t in trades if t.get("side") == "sell"]

        metrics.total_trades = min(len(buy_trades), len(sell_trades))

        if metrics.total_trades > 0:
            wins = 0
            total_profit = 0.0
            total_loss = 0.0

            # Calculate P&L for each round trip
            for i in range(metrics.total_trades):
                if i < len(buy_trades) and i < len(sell_trades):
                    buy = buy_trades[i]
                    sell = sell_trades[i]

                    try:
                        buy_price = float(buy.get("price", 0))
                        sell_price = float(sell.get("price", 0))
                        quantity = float(buy.get("quantity", 0))

                        if quantity > 0 and buy_price > 0 and sell_price > 0:
                            pnl = (sell_price - buy_price) * quantity

                            if pnl > 0:
                                wins += 1
                                total_profit += pnl
                            else:
                                total_loss += abs(pnl)

                    except (ValueError, TypeError) as e:
                        logger.warning(f"Invalid trade data at index {i}: {e}")
                        continue

            # Calculate win rate
            metrics.win_rate = (
                (wins / metrics.total_trades) * 100 if metrics.total_trades > 0 else 0.0
            )

            # Calculate profit factor
            metrics.profit_factor = total_profit / total_loss if total_loss > 0 else 0.0

        return metrics

    except Exception as e:
        logger.warning(f"Failed to calculate trade statistics: {e}")
        metrics.total_trades = 0
        metrics.win_rate = 0.0
        metrics.profit_factor = 0.0
        return metrics


def _calculate_sharpe_ratio(
    metrics: BacktestMetrics, returns: pd.Series, config: dict
) -> BacktestMetrics:
    """Calculate Sharpe ratio with safe division."""
    try:
        # Clean returns data
        returns_clean = returns.dropna()

        if len(returns_clean) <= 1:
            metrics.sharpe_ratio = 0.0
            return metrics

        # Get risk-free rate from config
        risk_free_rate = config.get("risk_free_rate", 0.02)  # Default 2% annual
        periods_per_year = config.get("periods_per_year", 252)  # Default 252 trading days

        # Convert annual risk-free rate to period rate
        rf_period = risk_free_rate / periods_per_year

        # Calculate excess returns
        excess_returns = returns_clean - rf_period

        # Calculate Sharpe ratio
        if len(excess_returns) > 1:
            mean_excess = float(excess_returns.mean())
            std_excess = float(excess_returns.std())

            if std_excess > 0 and not np.isnan(std_excess) and not np.isinf(std_excess):
                metrics.sharpe_ratio = (mean_excess / std_excess) * np.sqrt(periods_per_year)

                # Clamp extreme values
                if abs(metrics.sharpe_ratio) > 100:
                    logger.warning(f"Extreme Sharpe ratio calculated: {metrics.sharpe_ratio}")
                    metrics.sharpe_ratio = np.sign(metrics.sharpe_ratio) * 100
            else:
                metrics.sharpe_ratio = 0.0
        else:
            metrics.sharpe_ratio = 0.0

        return metrics

    except Exception as e:
        logger.warning(f"Failed to calculate Sharpe ratio: {e}")
        metrics.sharpe_ratio = 0.0
        return metrics


def _calculate_sortino_ratio(
    metrics: BacktestMetrics, returns: pd.Series, config: dict
) -> BacktestMetrics:
    """Calculate Sortino ratio (downside deviation)."""
    try:
        returns_clean = returns.dropna()

        if len(returns_clean) <= 1:
            metrics.sortino_ratio = 0.0
            return metrics

        # Get risk-free rate from config
        risk_free_rate = config.get("risk_free_rate", 0.02)
        periods_per_year = config.get("periods_per_year", 252)
        rf_period = risk_free_rate / periods_per_year

        # Calculate excess returns
        excess_returns = returns_clean - rf_period

        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) > 1:
            downside_std = float(downside_returns.std())
            mean_excess = float(excess_returns.mean())

            if downside_std > 0 and not np.isnan(downside_std):
                metrics.sortino_ratio = (mean_excess / downside_std) * np.sqrt(periods_per_year)

                # Clamp extreme values
                if abs(metrics.sortino_ratio) > 100:
                    metrics.sortino_ratio = np.sign(metrics.sortino_ratio) * 100
            else:
                metrics.sortino_ratio = 0.0
        else:
            # No downside volatility
            metrics.sortino_ratio = float("inf") if excess_returns.mean() > 0 else 0.0

        return metrics

    except Exception as e:
        logger.warning(f"Failed to calculate Sortino ratio: {e}")
        metrics.sortino_ratio = 0.0
        return metrics


def _calculate_calmar_ratio(
    metrics: BacktestMetrics, returns: pd.Series, equity_curve: pd.Series
) -> BacktestMetrics:
    """Calculate Calmar ratio (annual return / max drawdown)."""
    try:
        if not hasattr(metrics, "total_return") or not hasattr(metrics, "max_drawdown"):
            metrics.calmar_ratio = 0.0
            return metrics

        # Convert total return to annual return (assume returns are daily)
        if len(returns) > 0:
            total_days = len(returns)
            years = total_days / 252.0  # Approximate trading days per year

            if years > 0 and metrics.total_return is not None:
                annual_return = (((1 + metrics.total_return / 100) ** (1 / years)) - 1) * 100
            else:
                annual_return = metrics.total_return or 0
        else:
            annual_return = 0

        # Calculate Calmar ratio
        max_dd = getattr(metrics, "max_drawdown", 0) or 0
        if max_dd > 0:
            metrics.calmar_ratio = annual_return / max_dd
        else:
            metrics.calmar_ratio = 0.0

        return metrics

    except Exception as e:
        logger.warning(f"Failed to calculate Calmar ratio: {e}")
        metrics.calmar_ratio = 0.0
        return metrics


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers with fallback.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division fails

    Returns:
        Division result or default value
    """
    try:
        if denominator == 0 or np.isnan(denominator) or np.isinf(denominator):
            return default

        result = numerator / denominator

        if np.isnan(result) or np.isinf(result):
            return default

        return float(result)

    except (ZeroDivisionError, TypeError, ValueError):
        return default


def validate_metric_value(
    value: float, metric_name: str, bounds: tuple[float, float] | None = None
) -> float:
    """
    Validate and clamp metric values.

    Args:
        value: Metric value to validate
        metric_name: Name of the metric for logging
        bounds: Optional (min, max) bounds tuple

    Returns:
        Validated metric value
    """
    try:
        if pd.isna(value) or np.isinf(value):
            logger.warning(f"Invalid {metric_name}: {value}, setting to 0")
            return 0.0

        if bounds:
            min_val, max_val = bounds
            if value < min_val:
                logger.warning(f"{metric_name} below minimum: {value} < {min_val}")
                return min_val
            elif value > max_val:
                logger.warning(f"{metric_name} above maximum: {value} > {max_val}")
                return max_val

        return float(value)

    except (TypeError, ValueError):
        logger.warning(f"Could not convert {metric_name} to float: {value}")
        return 0.0
