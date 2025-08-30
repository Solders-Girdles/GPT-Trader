"""
Data fetching and validation for backtesting.
"""

from datetime import datetime
from typing import Optional
import pandas as pd
import yfinance as yf


def fetch_historical_data(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = '1d'
) -> pd.DataFrame:
    """
    Fetch historical market data for backtesting.
    
    Args:
        symbol: Stock symbol
        start: Start date
        end: End date  
        interval: Data interval
        
    Returns:
        DataFrame with OHLCV data
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start, end=end, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data available for {symbol}")
    
    # Standardize columns and convert to lowercase
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.columns = data.columns.str.lower()
    
    # Validate data using local validation
    from .validation import validate_data as validate_market_data
    if not validate_market_data(data):
        raise ValueError(f"Invalid data for {symbol}")
    
    return data


