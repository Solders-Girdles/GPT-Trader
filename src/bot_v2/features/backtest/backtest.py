"""
Main backtest orchestration - entry point for the slice.
"""

from datetime import datetime
from typing import Optional
import time
import logging

from .types import BacktestResult
from .data import fetch_historical_data
from .signals import generate_signals
from .execution import simulate_trades
from .metrics import calculate_metrics

# Import error handling and configuration
from ...errors import (
    BacktestError, DataError, ValidationError, StrategyError,
    handle_error, log_error as error_log
)
from ...errors.handler import get_error_handler, with_error_handling, RecoveryStrategy
from ...config import get_config
from ...validation import validate_inputs, StrategyNameValidator, SymbolValidator, DateValidator, PositiveNumberValidator, PercentageValidator

# Import logging from monitor slice with proper fallback
try:
    from ..monitor import log_event, log_performance, log_error, set_correlation_id, LogLevel
except ImportError:
    # Fallback if monitor slice not available
    logger = logging.getLogger(__name__)
    def log_event(event, message, **kwargs): 
        logger.info(f"{event}: {message}", extra=kwargs)
    def log_performance(operation, duration_ms, **kwargs): 
        logger.info(f"{operation} took {duration_ms:.2f}ms", extra=kwargs)
    def log_error(error, **kwargs): 
        logger.error(f"Error: {error}", extra=kwargs)
    def set_correlation_id(*args, **kwargs): pass
    LogLevel = None


@validate_inputs(
    strategy=StrategyNameValidator(),
    symbol=SymbolValidator(),
    start=DateValidator(),
    end=DateValidator(),
    initial_capital=PositiveNumberValidator(),
    commission=PercentageValidator(as_decimal=True),
    slippage=PercentageValidator(as_decimal=True)
)
def run_backtest(
    strategy: str,
    symbol: str,
    start: datetime,
    end: datetime,
    initial_capital: Optional[float] = None,
    commission: Optional[float] = None,
    slippage: Optional[float] = None,
    **strategy_params
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
    # Load configuration with defaults
    try:
        config = get_config('backtest')
    except Exception as e:
        raise BacktestError(
            "Failed to load backtest configuration",
            context={'error': str(e)}
        )
    
    # Apply config defaults
    initial_capital = initial_capital or config.get('initial_capital', 10000.0)
    commission = commission or config.get('commission', 0.001)
    slippage = slippage or config.get('slippage', 0.0005)
    
    # Additional validation
    if end <= start:
        raise ValidationError(
            "End date must be after start date",
            field="date_range",
            value=f"{start} to {end}"
        )
    
    # Validate minimum data requirements
    min_days = config.get('min_data_points', 30)
    date_diff = (end - start).days
    if date_diff < min_days:
        raise ValidationError(
            f"Date range too short: need at least {min_days} days, got {date_diff}",
            field="date_range",
            value=date_diff
        )
    # Set correlation ID for this backtest run
    set_correlation_id()
    
    # Log backtest start with error handling
    try:
        log_event(
            "backtest_start", 
            f"Starting backtest for {strategy} on {symbol}",
            level=LogLevel.INFO if LogLevel else None,
            strategy=strategy,
            symbol=symbol,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
    except Exception as e:
        # Don't fail backtest if logging fails
        logging.getLogger(__name__).warning(f"Failed to log backtest start: {e}")
    
    start_time = time.perf_counter()
    
    error_handler = get_error_handler()
    
    try:
        # Step 1: Fetch historical data with retry logic
        data_start = time.perf_counter()
        try:
            data = error_handler.with_retry(
                fetch_historical_data,
                symbol, start, end,
                recovery_strategy=RecoveryStrategy.RETRY
            )
        except Exception as e:
            raise DataError(
                f"Failed to fetch data for {symbol}",
                symbol=symbol,
                context={'start': start.isoformat(), 'end': end.isoformat()}
            ) from e
        
        data_time = (time.perf_counter() - data_start) * 1000
        log_performance("fetch_historical_data", data_time, success=True, rows=len(data))
        
        # Step 2: Generate trading signals with validation
        signals_start = time.perf_counter()
        try:
            signals = error_handler.with_retry(
                generate_signals,
                strategy, data,
                recovery_strategy=RecoveryStrategy.FAIL_FAST,
                **strategy_params
            )
        except Exception as e:
            raise StrategyError(
                f"Failed to generate signals using {strategy}",
                strategy_name=strategy,
                context={'data_rows': len(data), 'params': strategy_params}
            ) from e
        
        signals_time = (time.perf_counter() - signals_start) * 1000
        log_performance("generate_signals", signals_time, success=True, signal_count=len(signals))
        
        # Step 3: Simulate trade execution with validation
        trades_start = time.perf_counter()
        try:
            trades, equity_curve, returns = error_handler.with_retry(
                simulate_trades,
                signals, data, initial_capital, commission, slippage,
                recovery_strategy=RecoveryStrategy.FAIL_FAST
            )
        except Exception as e:
            raise BacktestError(
                "Failed to simulate trades",
                context={
                    'signals_count': len(signals),
                    'data_rows': len(data),
                    'initial_capital': initial_capital
                }
            ) from e
        
        trades_time = (time.perf_counter() - trades_start) * 1000
        log_performance("simulate_trades", trades_time, success=True, trade_count=len(trades))
        
        # Step 4: Calculate performance metrics with safe calculations
        metrics_start = time.perf_counter()
        try:
            metrics = error_handler.with_retry(
                calculate_metrics,
                trades, equity_curve, returns, initial_capital,
                recovery_strategy=RecoveryStrategy.DEGRADE
            )
        except Exception as e:
            raise BacktestError(
                "Failed to calculate performance metrics",
                context={
                    'trades_count': len(trades),
                    'equity_points': len(equity_curve),
                    'returns_points': len(returns)
                }
            ) from e
        
        metrics_time = (time.perf_counter() - metrics_start) * 1000
        log_performance("calculate_metrics", metrics_time, success=True)
        
        # Step 5: Validate and return complete results
        try:
            result = BacktestResult(
                trades=trades,
                equity_curve=equity_curve,
                returns=returns,
                metrics=metrics
            )
        except Exception as e:
            raise BacktestError(
                "Failed to create backtest result",
                context={
                    'trades_valid': trades is not None,
                    'equity_valid': equity_curve is not None,
                    'returns_valid': returns is not None,
                    'metrics_valid': metrics is not None
                }
            ) from e
        
        # Log successful completion with safe calculations
        total_time = (time.perf_counter() - start_time) * 1000
        
        try:
            final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
            total_return = ((final_equity - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0.0
            
            log_event(
                "backtest_complete",
                f"Backtest completed for {strategy} on {symbol}",
                level=LogLevel.INFO if LogLevel else None,
                strategy=strategy,
                symbol=symbol,
                duration_ms=total_time,
                total_trades=len(trades),
                final_equity=final_equity,
                total_return_pct=total_return,
                success=True
            )
        except Exception as e:
            # Don't fail backtest if final logging fails
            logging.getLogger(__name__).warning(f"Failed to log completion: {e}")
        
        return result
        
    except (BacktestError, DataError, ValidationError, StrategyError) as e:
        # Handle known trading errors with proper context
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Add backtest context to error
        e.add_context(
            strategy=strategy,
            symbol=symbol,
            duration_ms=total_time,
            start_date=start.isoformat(),
            end_date=end.isoformat()
        )
        
        # Log the error
        error_log(e)
        
        try:
            log_event(
                "backtest_failed",
                f"Backtest failed for {strategy} on {symbol}: {e.message}",
                level=LogLevel.ERROR if LogLevel else None,
                strategy=strategy,
                symbol=symbol,
                error_code=e.error_code,
                error_message=e.message,
                duration_ms=total_time,
                recoverable=e.recoverable
            )
        except Exception:
            # Don't fail if logging fails
            pass
        
        # Re-raise the trading error
        raise
    
    except Exception as e:
        # Handle unexpected errors
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Wrap in BacktestError
        backtest_error = BacktestError(
            f"Unexpected error during backtest: {str(e)}",
            context={
                'strategy': strategy,
                'symbol': symbol,
                'duration_ms': total_time,
                'original_error': str(e),
                'error_type': type(e).__name__
            }
        )
        
        # Log the wrapped error
        error_log(backtest_error)
        
        try:
            log_event(
                "backtest_failed",
                f"Backtest failed for {strategy} on {symbol}: unexpected error",
                level=LogLevel.ERROR if LogLevel else None,
                strategy=strategy,
                symbol=symbol,
                error_type=type(e).__name__,
                error_message=str(e),
                duration_ms=total_time
            )
        except Exception:
            # Don't fail if logging fails
            pass
        
        # Re-raise the wrapped error
        raise backtest_error
