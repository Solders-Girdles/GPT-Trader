"""Examples of using the enhanced exception system.

This module demonstrates best practices for using the exception
hierarchy and decorators in real trading scenarios.
"""

from __future__ import annotations

import logging
import random
import time

import numpy as np
import pandas as pd

from .decorators import (
    handle_exceptions,
    monitor_performance,
    safe_execution,
    validate_inputs,
    with_circuit_breaker,
    with_recovery,
    with_retry,
)
from .enhanced_exceptions import (
    CriticalError,
    DataIntegrityError,
    InsufficientCapitalError,
    NetworkError,
    OrderRejectedError,
    RecoverableError,
    RetryableError,
    RiskLimitError,
    get_exception_handler,
)

logger = logging.getLogger(__name__)


# Example 1: Data fetching with retry and circuit breaker
@with_retry(max_retries=3, backoff_base=2.0, exceptions=(NetworkError,))
@with_circuit_breaker(failure_threshold=5, timeout_seconds=60)
def fetch_market_data(symbol: str) -> pd.DataFrame:
    """Fetch market data with automatic retry and circuit breaking.

    Args:
        symbol: Stock symbol to fetch.

    Returns:
        DataFrame with market data.

    Raises:
        NetworkError: If network request fails.
    """
    # Simulate network request
    if random.random() < 0.2:  # 20% chance of failure
        raise NetworkError(
            f"Failed to fetch data for {symbol}",
            url=f"https://api.market.com/data/{symbol}",
            status_code=500,
        )

    # Return dummy data
    return pd.DataFrame(
        {
            "Open": np.random.randn(100) + 100,
            "High": np.random.randn(100) + 101,
            "Low": np.random.randn(100) + 99,
            "Close": np.random.randn(100) + 100,
            "Volume": np.random.randint(1000000, 10000000, 100),
        }
    )


# Example 2: Data validation with recovery
@with_recovery(
    fallback_function=lambda df: df.fillna(method="ffill").fillna(method="bfill"),
    component="data_validation",
    operation="clean_ohlc",
)
def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean OHLC data with automatic recovery.

    Args:
        df: DataFrame to validate.

    Returns:
        Cleaned DataFrame.

    Raises:
        DataIntegrityError: If data cannot be repaired.
    """
    # Check for NaN values
    if df.isna().any().any():
        # Attempt to repair
        def repair_data(data):
            return data.fillna(method="ffill").fillna(method="bfill")

        raise DataIntegrityError("Data contains NaN values", data=df, repair_function=repair_data)

    # Check OHLC relationships
    invalid_rows = df["High"] < df["Low"]
    if invalid_rows.any():
        raise DataIntegrityError(
            f"Invalid OHLC relationships in {invalid_rows.sum()} rows", data=df
        )

    return df


# Example 3: Order execution with comprehensive error handling
@handle_exceptions(
    recoverable_exceptions=(InsufficientCapitalError,),
    retryable_exceptions=(OrderRejectedError,),
    critical_exceptions=(RiskLimitError,),
    fallback_value=None,
    component="order_execution",
    max_retries=2,
)
def execute_order(
    symbol: str, quantity: int, price: float, available_capital: float, risk_limit: float
) -> dict | None:
    """Execute a trading order with comprehensive error handling.

    Args:
        symbol: Symbol to trade.
        quantity: Number of shares.
        price: Price per share.
        available_capital: Available capital.
        risk_limit: Maximum risk allowed.

    Returns:
        Order confirmation or None.

    Raises:
        Various trading exceptions.
    """
    required_capital = quantity * price

    # Check capital
    if required_capital > available_capital:
        raise InsufficientCapitalError(
            f"Insufficient capital for {symbol}",
            required_capital=required_capital,
            available_capital=available_capital,
            symbol=symbol,
        )

    # Check risk limit
    position_risk = required_capital * 0.02  # 2% position risk
    if position_risk > risk_limit:
        raise RiskLimitError(
            f"Risk limit exceeded for {symbol}",
            limit_type="position_risk",
            limit_value=risk_limit,
            current_value=position_risk,
        )

    # Simulate order submission
    if random.random() < 0.1:  # 10% chance of rejection
        raise OrderRejectedError(
            f"Order rejected for {symbol}",
            order_id=f"ORD-{random.randint(1000, 9999)}",
            reason="Insufficient liquidity",
            symbol=symbol,
        )

    # Return confirmation
    return {
        "order_id": f"ORD-{random.randint(1000, 9999)}",
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "status": "filled",
    }


# Example 4: Safe calculation with validation
@safe_execution(default_return=0.0, log_errors=True)
@validate_inputs(
    returns=lambda x: isinstance(x, list | np.ndarray | pd.Series),
    risk_free_rate=lambda x: 0 <= x <= 1,
)
def calculate_sharpe_ratio(returns, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio with input validation and safe execution.

    Args:
        returns: Return series.
        risk_free_rate: Risk-free rate.

    Returns:
        Sharpe ratio or 0.0 on error.
    """
    returns = np.array(returns)

    if len(returns) < 2:
        raise ValueError("Insufficient data for Sharpe ratio calculation")

    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate

    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


# Example 5: Performance monitoring
@monitor_performance(slow_threshold_seconds=0.5, component="backtest")
def run_backtest_with_monitoring(strategy, data: pd.DataFrame) -> dict:
    """Run backtest with performance monitoring.

    Args:
        strategy: Strategy to test.
        data: Historical data.

    Returns:
        Backtest results.
    """
    # Simulate backtest
    time.sleep(random.uniform(0.1, 0.8))  # Simulate processing time

    return {
        "total_return": random.uniform(-0.1, 0.3),
        "sharpe_ratio": random.uniform(0.5, 2.0),
        "max_drawdown": random.uniform(0.05, 0.25),
        "trades": random.randint(50, 200),
    }


# Example 6: Using the exception handler directly
def comprehensive_trading_example():
    """Example of comprehensive exception handling in a trading workflow."""
    handler = get_exception_handler()

    # Simulate trading workflow
    symbols = ["AAPL", "GOOGL", "MSFT"]
    results = []

    for symbol in symbols:
        try:
            # Fetch data
            logger.info(f"Processing {symbol}")
            data = fetch_market_data(symbol)

            # Validate data
            clean_data = validate_and_clean_data(data)

            # Calculate metrics
            returns = clean_data["Close"].pct_change().dropna()
            sharpe = calculate_sharpe_ratio(returns)

            # Execute order if conditions met
            if sharpe > 1.0:
                order = execute_order(
                    symbol=symbol,
                    quantity=100,
                    price=float(clean_data["Close"].iloc[-1]),
                    available_capital=10000.0,
                    risk_limit=200.0,
                )
                results.append({"symbol": symbol, "sharpe": sharpe, "order": order})

        except RecoverableError as e:
            # Handle with recovery
            result = handler.handle(e)
            logger.info(f"Recovered from error for {symbol}: {result}")

        except RetryableError as e:
            # Handle with retry
            result = handler.handle(e)
            if result == "retry":
                logger.info(f"Retrying {symbol}")
                continue

        except CriticalError as e:
            # Handle critical error
            handler.handle(e)
            logger.critical("Critical error, stopping processing")
            break

        except Exception as e:
            logger.error(f"Unexpected error for {symbol}: {e}")
            continue

    # Get statistics
    stats = handler.get_stats()
    logger.info(f"Exception handling stats: {stats}")

    return results


# Example 7: Custom recovery action
def custom_recovery_example():
    """Example of custom recovery actions."""

    def fetch_from_cache(symbol: str) -> pd.DataFrame:
        """Fallback to cached data."""
        logger.info(f"Fetching {symbol} from cache")
        # Return cached data
        return pd.DataFrame({"Close": np.random.randn(50) + 100})

    def fetch_with_fallback(symbol: str) -> pd.DataFrame:
        """Fetch data with custom fallback."""
        try:
            # Try primary source
            return fetch_market_data(symbol)
        except NetworkError:
            # Create recoverable error with custom recovery
            error = RecoverableError(
                f"Failed to fetch {symbol}, using cache",
                recovery_action=lambda: fetch_from_cache(symbol),
                component="data_fetch",
            )

            # Attempt recovery
            handler = get_exception_handler()
            return handler.handle(error)

    # Use the function
    data = fetch_with_fallback("AAPL")
    return data


if __name__ == "__main__":
    # Run examples
    logging.basicConfig(level=logging.INFO)

    print("Running comprehensive trading example...")
    results = comprehensive_trading_example()
    print(f"Results: {results}")

    print("\nRunning custom recovery example...")
    data = custom_recovery_example()
    print(f"Data shape: {data.shape}")
