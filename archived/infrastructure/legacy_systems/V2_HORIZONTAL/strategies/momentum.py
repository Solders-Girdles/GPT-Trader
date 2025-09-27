"""
Momentum strategy based on rate of change.
Captures price momentum and trend continuation patterns.
"""

import pandas as pd
import numpy as np
from .base import StrategyBase, StrategyConfig


class MomentumStrategy(StrategyBase):
    """
    Momentum strategy using rate of change (ROC) indicator.
    
    Buys when momentum is strongly positive and accelerating.
    Sells when momentum turns negative or starts weakening.
    
    Logic:
    - Calculate ROC over lookback period
    - Buy when ROC > buy_threshold and increasing
    - Sell when ROC < sell_threshold or decreasing rapidly
    """
    
    def __init__(
        self, 
        lookback_period: int = 14,
        buy_threshold: float = 2.0,
        sell_threshold: float = -1.0,
        momentum_smoothing: int = 3
    ):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_period: Period for rate of change calculation
            buy_threshold: ROC percentage threshold for buy signals
            sell_threshold: ROC percentage threshold for sell signals  
            momentum_smoothing: Period for smoothing momentum signals
        """
        config = StrategyConfig(
            name="Momentum",
            description=f"ROC momentum strategy ({lookback_period}d lookback, "
                       f"{buy_threshold}%/{sell_threshold}% thresholds)",
            parameters={
                'lookback_period': lookback_period,
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'momentum_smoothing': momentum_smoothing
            }
        )
        super().__init__(config)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate momentum-based trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of signals: 1 = buy, -1 = sell, 0 = hold
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must have 'Close' column")
        
        # Get parameters
        lookback = self.get_parameter('lookback_period')
        buy_threshold = self.get_parameter('buy_threshold')
        sell_threshold = self.get_parameter('sell_threshold')
        smoothing = self.get_parameter('momentum_smoothing')
        
        # Calculate rate of change (momentum)
        roc = data['Close'].pct_change(periods=lookback) * 100
        
        # Smooth the momentum signal
        roc_smooth = roc.rolling(window=smoothing).mean()
        
        # Calculate momentum change (acceleration/deceleration)
        roc_change = roc_smooth.diff()
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        for i in range(smoothing + lookback, len(data)):
            current_roc = roc_smooth.iloc[i]
            roc_delta = roc_change.iloc[i]
            
            # Skip if we don't have valid data
            if pd.isna(current_roc) or pd.isna(roc_delta):
                continue
            
            # Buy signal: Strong positive momentum that's accelerating
            if current_roc > buy_threshold and roc_delta > 0:
                signals.iloc[i] = 1
                
            # Sell signal: Negative momentum or rapidly decelerating positive momentum
            elif current_roc < sell_threshold or (current_roc > 0 and roc_delta < -0.5):
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
        
        # Check for non-positive prices (would break percentage calculations)
        if (data['Close'] <= 0).any():
            return False
        
        return True
    
    def get_required_periods(self) -> int:
        """Get minimum number of periods required for signal generation."""
        lookback = self.get_parameter('lookback_period')
        smoothing = self.get_parameter('momentum_smoothing')
        # Need lookback + smoothing + 1 for momentum change calculation
        return lookback + smoothing + 2