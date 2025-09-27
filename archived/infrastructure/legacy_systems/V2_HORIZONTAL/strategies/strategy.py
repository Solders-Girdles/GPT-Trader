"""
Simple moving average crossover strategy.
Built on the StrategyBase foundation for consistency.
"""

import pandas as pd
import numpy as np
from typing import Dict
from .base import StrategyBase, StrategyConfig


class SimpleMAStrategy(StrategyBase):
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
        config = StrategyConfig(
            name="SimpleMA",
            description=f"MA crossover strategy ({fast_period}/{slow_period})",
            parameters={
                'fast_period': fast_period,
                'slow_period': slow_period
            }
        )
        super().__init__(config)
        
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
        
        # Get parameters from config
        fast_period = self.get_parameter('fast_period')
        slow_period = self.get_parameter('slow_period')
        
        # Calculate moving averages
        fast_ma = data['Close'].rolling(window=fast_period).mean()
        slow_ma = data['Close'].rolling(window=slow_period).mean()
        
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
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data meets strategy requirements."""
        # Check for required columns
        if 'Close' not in data.columns:
            return False
        
        # Check for sufficient data
        required_periods = self.get_required_periods()
        if len(data) < required_periods:
            return False
        
        # Check for null values in Close column
        if data['Close'].isna().any():
            return False
            
        return True
    
    def get_required_periods(self) -> int:
        """Get minimum number of periods required for signal generation."""
        fast_period = self.get_parameter('fast_period')
        slow_period = self.get_parameter('slow_period')
        # Need enough data for the slower MA plus one period for crossover detection
        return max(fast_period, slow_period) + 1