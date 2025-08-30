"""
Minimal strategy module - ONE simple MA crossover strategy.
Clear signals: 1 = buy, -1 = sell, 0 = hold.
"""

import pandas as pd
import numpy as np
from typing import Dict


class SimpleMAStrategy:
    """
    Simple moving average crossover strategy.
    Buy when fast MA crosses above slow MA.
    Sell when fast MA crosses below slow MA.
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        """
        Args:
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MA crossover.
        
        Args:
            data: DataFrame with at least a 'Close' column
            
        Returns:
            Series of signals: 1 = buy, -1 = sell, 0 = hold
            Index matches the input data index
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must have 'Close' column")
        
        # Calculate moving averages
        fast_ma = data['Close'].rolling(window=self.fast_period).mean()
        slow_ma = data['Close'].rolling(window=self.slow_period).mean()
        
        # Initialize signals to 0 (hold)
        signals = pd.Series(0, index=data.index)
        
        # Generate crossover signals
        # Current fast > slow AND previous fast <= slow = BUY
        # Current fast < slow AND previous fast >= slow = SELL
        
        for i in range(1, len(data)):
            # Skip if MAs not yet calculated
            if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]):
                continue
                
            # Check for crossover
            curr_fast = fast_ma.iloc[i]
            curr_slow = slow_ma.iloc[i]
            prev_fast = fast_ma.iloc[i-1]
            prev_slow = slow_ma.iloc[i-1]
            
            # Buy signal: fast crosses above slow
            if curr_fast > curr_slow and prev_fast <= prev_slow:
                signals.iloc[i] = 1
            # Sell signal: fast crosses below slow  
            elif curr_fast < curr_slow and prev_fast >= prev_slow:
                signals.iloc[i] = -1
                
        return signals
    
    def get_parameters(self) -> Dict:
        """Return current strategy parameters."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period
        }