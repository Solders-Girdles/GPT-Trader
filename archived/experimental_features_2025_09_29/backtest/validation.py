"""
Local validation utilities for backtesting.

Self-contained validation functions with no external dependencies.
"""

import pandas as pd


def validate_data(data: pd.DataFrame) -> bool:
    """
    Validate market data for backtesting.

    Args:
        data: DataFrame with OHLCV data

    Returns:
        True if data is valid
    """
    # Check required columns
    required_columns = ["open", "high", "low", "close", "volume"]
    if not all(col in data.columns for col in required_columns):
        return False

    # Check for nulls
    if data[required_columns].isnull().any().any():
        return False

    # Check OHLC relationships
    if not (data["high"] >= data["low"]).all():
        return False
    if not (data["high"] >= data["open"]).all():
        return False
    if not (data["high"] >= data["close"]).all():
        return False
    if not (data["low"] <= data["open"]).all():
        return False
    if not (data["low"] <= data["close"]).all():
        return False

    # Check for negative prices
    if (data[["open", "high", "low", "close"]] < 0).any().any():
        return False

    # Check for negative volume
    if (data["volume"] < 0).any():
        return False

    return True


def validate_signals(signals: pd.Series) -> bool:
    """
    Validate trading signals.

    Args:
        signals: Series of signals to validate

    Returns:
        True if signals are valid
    """
    # Check for valid values
    valid_values = {-1, 0, 1}
    unique_values = set(signals.dropna().unique())
    if not unique_values.issubset(valid_values):
        return False

    # Check signal rate (basic sanity check)
    signal_rate = (signals != 0).mean()
    if signal_rate > 0.8:  # More than 80% signals seems excessive
        return False

    return True


def validate_parameters(
    initial_capital: float, commission: float, slippage: float
) -> tuple[bool, str | None]:
    """
    Validate backtest parameters.

    Args:
        initial_capital: Starting capital
        commission: Commission rate
        slippage: Slippage rate

    Returns:
        (is_valid, error_message)
    """
    if initial_capital <= 0:
        return False, "Initial capital must be positive"

    if commission < 0 or commission > 0.1:  # Max 10% commission
        return False, "Commission must be between 0 and 0.1"

    if slippage < 0 or slippage > 0.1:  # Max 10% slippage
        return False, "Slippage must be between 0 and 0.1"

    return True, None


def validate_trades(trades: pd.DataFrame) -> bool:
    """
    Validate trade execution results.

    Args:
        trades: DataFrame of executed trades

    Returns:
        True if trades are valid
    """
    if trades.empty:
        return True  # No trades is valid

    required_columns = [
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "position_size",
        "pnl",
    ]

    if not all(col in trades.columns for col in required_columns):
        return False

    # Check dates
    if not (trades["exit_date"] >= trades["entry_date"]).all():
        return False

    # Check prices
    if (trades[["entry_price", "exit_price"]] <= 0).any().any():
        return False

    return True
