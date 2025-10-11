"""
Main backtest orchestration - entry point for the slice.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from bot_v2.features.monitor import LogLevel as LogLevelType
else:

    class LogLevelType(Enum):
        DEBUG = auto()
        INFO = auto()
        WARNING = auto()
        ERROR = auto()


import pandas as pd
from bot_v2.config import get_config

# Import error handling and configuration
from bot_v2.errors import BacktestError, DataError, StrategyError, TradingError, ValidationError
from bot_v2.errors import log_error as error_log
from bot_v2.errors.handler import RecoveryStrategy, get_error_handler
from bot_v2.features.backtest.data import fetch_historical_data
from bot_v2.features.backtest.execution import simulate_trades
from bot_v2.features.backtest.metrics import calculate_metrics
from bot_v2.features.backtest.signals import generate_signals
from bot_v2.features.backtest.types import BacktestMetrics, BacktestResult, TradeDict
from bot_v2.validation import (
    DateValidator,
    PercentageValidator,
    PositiveNumberValidator,
    StrategyNameValidator,
    SymbolValidator,
    validate_inputs,
)

# Import logging from monitor slice with proper fallback
try:
    from bot_v2.features.monitor import log_error, log_event, log_performance, set_correlation_id
except ImportError:
    # Fallback if monitor slice not available
    logger = logging.getLogger(__name__)

    LogLevel = LogLevelType

    def log_event(
        event_type: str, message: str, level: LogLevelType = LogLevelType.INFO, **kwargs: Any
    ) -> None:
        logger.info(
            f"{event_type}: {message}",
            extra={"level": level.name, **kwargs},
        )

    def log_performance(operation: str, duration_ms: float, **kwargs: Any) -> None:
        logger.info(f"{operation} took {duration_ms:.2f}ms", extra=kwargs)

    def log_error(error: Exception, context: str | None = None, **kwargs: Any) -> None:
        logger.error(
            f"Error: {error}",
            extra={"context": context, **kwargs},
        )

    def set_correlation_id(correlation_id: str | None = None) -> None:
        pass


@dataclass
class BacktestConfigContext:
    initial_capital: float
    commission: float
    slippage: float


@validate_inputs(
    strategy=StrategyNameValidator(),
    symbol=SymbolValidator(),
    start=DateValidator(),
    end=DateValidator(),
    initial_capital=PositiveNumberValidator(),
    commission=PercentageValidator(as_decimal=True),
    slippage=PercentageValidator(as_decimal=True),
)
def run_backtest(
    strategy: str,
    symbol: str,
    start: datetime,
    end: datetime,
    initial_capital: float | None = None,
    commission: float | None = None,
    slippage: float | None = None,
    **strategy_params: Any,
) -> BacktestResult:
    """
    Run a complete backtest.

    This is the main entry point for the backtest feature slice.
    Everything needed for backtesting is contained within this slice.

    Args:
        strategy: Name of strategy to test
        symbol: Stock symbol
        start: Start date
        end: End date
        initial_capital: Starting capital (uses config default if None)
        commission: Commission rate (uses config default if None)
        slippage: Slippage rate (uses config default if None)
        **strategy_params: Additional strategy parameters

    Returns:
        BacktestResult with trades, metrics, and equity curve

    Raises:
        BacktestError: If backtest fails
        ValidationError: If parameters are invalid
        DataError: If data issues occur
        StrategyError: If strategy execution fails
    """
    config_context = _prepare_backtest_config(
        strategy=strategy,
        symbol=symbol,
        start=start,
        end=end,
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
    )

    initial_capital = config_context.initial_capital
    commission = config_context.commission
    slippage = config_context.slippage

    start_time = time.perf_counter()
    error_handler = get_error_handler()

    try:
        data = _fetch_market_data(
            error_handler=error_handler,
            symbol=symbol,
            start=start,
            end=end,
        )

        signals = _generate_trading_signals(
            error_handler=error_handler,
            strategy=strategy,
            data=data,
            strategy_params=strategy_params,
        )

        trades, equity_curve, returns = _execute_trade_simulation(
            error_handler=error_handler,
            data=data,
            signals=signals,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
        )

        metrics = _calculate_performance_metrics(
            error_handler=error_handler,
            trades=trades,
            equity_curve=equity_curve,
            returns=returns,
            initial_capital=initial_capital,
        )

        result = _assemble_backtest_result(
            trades=trades,
            equity_curve=equity_curve,
            returns=returns,
            metrics=metrics,
            initial_capital=initial_capital,
        )

        _log_backtest_completion(
            strategy=strategy,
            symbol=symbol,
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            start_time=start_time,
        )

        return result

    except (BacktestError, DataError, ValidationError, StrategyError) as exc:
        _handle_known_backtest_error(
            error=exc,
            strategy=strategy,
            symbol=symbol,
            start_date=start,
            end_date=end,
            start_time=start_time,
        )
        raise
    except Exception as exc:
        _handle_unexpected_backtest_error(
            error=exc,
            strategy=strategy,
            symbol=symbol,
            start_date=start,
            end_date=end,
            start_time=start_time,
        )
        raise


def _prepare_backtest_config(
    *,
    strategy: str,
    symbol: str,
    start: datetime,
    end: datetime,
    initial_capital: float | None,
    commission: float | None,
    slippage: float | None,
) -> BacktestConfigContext:
    try:
        config = get_config("backtest")
    except Exception as exc:
        raise BacktestError(
            "Failed to load backtest configuration",
            context={"error": str(exc)},
        ) from exc

    resolved_initial_capital = initial_capital or config.get("initial_capital", 10000.0)
    resolved_commission = commission or config.get("commission", 0.001)
    resolved_slippage = slippage or config.get("slippage", 0.0005)

    if end <= start:
        raise ValidationError(
            "End date must be after start date",
            field="date_range",
            value=f"{start} to {end}",
        )

    min_days = config.get("min_data_points", 30)
    date_diff = (end - start).days
    if date_diff < min_days:
        raise ValidationError(
            f"Date range too short: need at least {min_days} days, got {date_diff}",
            field="date_range",
            value=date_diff,
        )

    set_correlation_id()

    try:
        log_event(
            "backtest_start",
            f"Starting backtest for {strategy} on {symbol}",
            strategy=strategy,
            symbol=symbol,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            initial_capital=resolved_initial_capital,
            commission=resolved_commission,
            slippage=resolved_slippage,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(f"Failed to log backtest start: {exc}")

    return BacktestConfigContext(
        initial_capital=resolved_initial_capital,
        commission=resolved_commission,
        slippage=resolved_slippage,
    )


def _fetch_market_data(
    *,
    error_handler: Any,
    symbol: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    data_start = time.perf_counter()
    try:
        data = cast(
            pd.DataFrame,
            error_handler.with_retry(
                fetch_historical_data,
                symbol,
                start,
                end,
                recovery_strategy=RecoveryStrategy.RETRY,
            ),
        )
    except Exception as exc:
        raise DataError(
            f"Failed to fetch data for {symbol}",
            symbol=symbol,
            context={"start": start.isoformat(), "end": end.isoformat()},
        ) from exc

    data_time = (time.perf_counter() - data_start) * 1000
    log_performance("fetch_historical_data", data_time, success=True, rows=len(data))
    return data


def _generate_trading_signals(
    *,
    error_handler: Any,
    strategy: str,
    data: pd.DataFrame,
    strategy_params: dict[str, Any],
) -> pd.Series:
    signals_start = time.perf_counter()
    try:
        signals = cast(
            pd.Series,
            error_handler.with_retry(
                generate_signals,
                strategy,
                data,
                recovery_strategy=RecoveryStrategy.FAIL_FAST,
                **strategy_params,
            ),
        )
    except Exception as exc:
        raise StrategyError(
            f"Failed to generate signals using {strategy}",
            strategy_name=strategy,
            context={"data_rows": len(data), "params": strategy_params},
        ) from exc

    signals_time = (time.perf_counter() - signals_start) * 1000
    log_performance(
        "generate_signals",
        signals_time,
        success=True,
        signal_count=len(signals),
    )
    return signals


def _execute_trade_simulation(
    *,
    error_handler: Any,
    data: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float,
    commission: float,
    slippage: float,
) -> tuple[list[TradeDict], pd.Series, pd.Series]:
    trades_start = time.perf_counter()
    try:
        trades, equity_curve, returns = cast(
            tuple[list[TradeDict], pd.Series, pd.Series],
            error_handler.with_retry(
                simulate_trades,
                signals,
                data,
                initial_capital,
                commission,
                slippage,
                recovery_strategy=RecoveryStrategy.FAIL_FAST,
            ),
        )
    except Exception as exc:
        raise BacktestError(
            "Failed to simulate trades",
            context={
                "signals_count": len(signals),
                "data_rows": len(data),
                "initial_capital": initial_capital,
            },
        ) from exc

    trades_time = (time.perf_counter() - trades_start) * 1000
    log_performance(
        "simulate_trades",
        trades_time,
        success=True,
        trade_count=len(trades),
    )
    return trades, equity_curve, returns


def _calculate_performance_metrics(
    *,
    error_handler: Any,
    trades: list[TradeDict],
    equity_curve: pd.Series,
    returns: pd.Series,
    initial_capital: float,
) -> BacktestMetrics:
    metrics_start = time.perf_counter()
    try:
        metrics = cast(
            BacktestMetrics,
            error_handler.with_retry(
                calculate_metrics,
                trades,
                equity_curve,
                returns,
                initial_capital,
                recovery_strategy=RecoveryStrategy.DEGRADE,
            ),
        )
    except Exception as exc:
        raise BacktestError(
            "Failed to calculate performance metrics",
            context={
                "trades_count": len(trades),
                "equity_points": len(equity_curve),
                "returns_points": len(returns),
            },
        ) from exc

    metrics_time = (time.perf_counter() - metrics_start) * 1000
    log_performance("calculate_metrics", metrics_time, success=True)
    return metrics


def _assemble_backtest_result(
    *,
    trades: list[TradeDict],
    equity_curve: pd.Series,
    returns: pd.Series,
    metrics: BacktestMetrics,
    initial_capital: float,
) -> BacktestResult:
    try:
        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            returns=returns,
            metrics=metrics,
            initial_capital=initial_capital,
        )
    except Exception as exc:
        raise BacktestError(
            "Failed to create backtest result",
            context={
                "trades_valid": trades is not None,
                "equity_valid": equity_curve is not None,
                "returns_valid": returns is not None,
                "metrics_valid": metrics is not None,
            },
        ) from exc


def _log_backtest_completion(
    *,
    strategy: str,
    symbol: str,
    trades: list[TradeDict],
    equity_curve: pd.Series,
    initial_capital: float,
    start_time: float,
) -> None:
    total_time = (time.perf_counter() - start_time) * 1000
    try:
        final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
        total_return = (
            ((final_equity - initial_capital) / initial_capital) * 100
            if initial_capital > 0
            else 0.0
        )
        log_event(
            "backtest_complete",
            f"Backtest completed for {strategy} on {symbol}",
            strategy=strategy,
            symbol=symbol,
            duration_ms=total_time,
            total_trades=len(trades),
            final_equity=final_equity,
            total_return_pct=total_return,
            success=True,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(f"Failed to log completion: {exc}")


def _handle_known_backtest_error(
    *,
    error: TradingError,
    strategy: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    start_time: float,
) -> None:
    total_time = (time.perf_counter() - start_time) * 1000

    error.add_context(
        strategy=strategy,
        symbol=symbol,
        duration_ms=total_time,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    error_log(error)

    try:
        log_event(
            "backtest_failed",
            f"Backtest failed for {strategy} on {symbol}: {error.message}",
            strategy=strategy,
            symbol=symbol,
            error_code=getattr(error, "error_code", None),
            error_message=getattr(error, "message", str(error)),
            duration_ms=total_time,
            recoverable=getattr(error, "recoverable", False),
        )
    except Exception:
        pass

    raise error


def _handle_unexpected_backtest_error(
    *,
    error: Exception,
    strategy: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    start_time: float,
) -> None:
    total_time = (time.perf_counter() - start_time) * 1000

    backtest_error = BacktestError(
        f"Unexpected error during backtest: {str(error)}",
        context={
            "strategy": strategy,
            "symbol": symbol,
            "duration_ms": total_time,
            "original_error": str(error),
            "error_type": type(error).__name__,
        },
    )

    error_log(backtest_error)

    try:
        log_event(
            "backtest_failed",
            f"Backtest failed for {strategy} on {symbol}: unexpected error",
            strategy=strategy,
            symbol=symbol,
            error_type=type(error).__name__,
            error_message=str(error),
            duration_ms=total_time,
        )
    except Exception:
        pass

    raise backtest_error
