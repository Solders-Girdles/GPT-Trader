"""
Minimal data module - just YFinance, nothing else.
Clear, simple, testable.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional


class DataLoader:
    """Simple data loader that just gets OHLCV data."""
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        
    def get_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a symbol.
        
        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index is DatetimeIndex
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d',
                auto_adjust=True  # Adjust for splits/dividends
            )
            
            if df.empty:
                print(f"No data found for {symbol}")
                return None
            
            # Keep only OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Cache it
            self.cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return None
    
    def clear_cache(self):
        """Clear the cache."""
        self.cache = {}