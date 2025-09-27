"""
Breakout strategy using support/resistance levels and volume confirmation.
Trades on price breakouts from consolidation patterns.
"""

import pandas as pd
import numpy as np
from .base import StrategyBase, StrategyConfig


class BreakoutStrategy(StrategyBase):
    """
    Breakout strategy using price levels and volume.
    
    Buys when price breaks above resistance with volume.
    Sells when price breaks below support or fails to hold breakout.
    
    Logic:
    - Identify resistance as rolling maximum over lookback period
    - Identify support as rolling minimum over lookback period
    - Buy when price closes above resistance with volume surge
    - Sell when price closes below support or retraces from breakout
    """
    
    def __init__(
        self, 
        lookback_period: int = 20,
        breakout_threshold: float = 0.5,
        volume_multiplier: float = 1.5,
        retracement_threshold: float = 2.0,
        confirmation_periods: int = 2
    ):
        """
        Initialize breakout strategy.
        
        Args:
            lookback_period: Period for identifying support/resistance levels
            breakout_threshold: Percentage above resistance to confirm breakout
            volume_multiplier: Required volume increase for valid breakout
            retracement_threshold: Percentage retracement to trigger sell
            confirmation_periods: Number of periods to confirm breakout
        """
        config = StrategyConfig(
            name="Breakout",
            description=f"Breakout strategy ({lookback_period}d levels, "
                       f"{breakout_threshold}% threshold, {volume_multiplier}x volume)",
            parameters={
                'lookback_period': lookback_period,
                'breakout_threshold': breakout_threshold,
                'volume_multiplier': volume_multiplier,
                'retracement_threshold': retracement_threshold,
                'confirmation_periods': confirmation_periods
            }
        )
        super().__init__(config)
    
    def _identify_levels(self, data: pd.DataFrame, lookback: int) -> tuple:
        """Identify support and resistance levels."""
        # Resistance: Rolling maximum high, shifted to avoid including current period
        # We shift by 2 to ensure we're looking at historical resistance
        resistance = data['High'].rolling(window=lookback).max().shift(2)
        
        # Support: Rolling minimum low, shifted to avoid including current period
        support = data['Low'].rolling(window=lookback).min().shift(2)
        
        return support, resistance
    
    def _calculate_volume_surge(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume surge relative to average."""
        volume_ma = data['Volume'].rolling(window=20).mean()
        volume_surge = data['Volume'] / volume_ma
        return volume_surge
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate breakout trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of signals: 1 = buy, -1 = sell, 0 = hold
        """
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must have '{col}' column")
        
        # Get parameters
        lookback = self.get_parameter('lookback_period')
        breakout_threshold = self.get_parameter('breakout_threshold')
        volume_multiplier = self.get_parameter('volume_multiplier')
        retracement_threshold = self.get_parameter('retracement_threshold')
        
        # Calculate volume surge
        volume_surge = self._calculate_volume_surge(data)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        in_position = False
        entry_price = None
        
        # Start from minimum required period
        min_period = lookback + 20  # Need lookback + volume MA period
        
        for i in range(min_period, len(data)):
            current_close = data['Close'].iloc[i]
            current_volume_surge = volume_surge.iloc[i] if not pd.isna(volume_surge.iloc[i]) else 1.0
            
            # Calculate recent high and low for this specific point
            recent_high = data['High'].iloc[i-lookback:i].max()
            recent_low = data['Low'].iloc[i-lookback:i].min()
            
            if pd.isna(recent_high) or pd.isna(recent_low):
                continue
            
            # Define breakout threshold
            breakout_level = recent_high * (1 + breakout_threshold / 100)
            breakdown_level = recent_low * (1 - breakout_threshold / 100)
            
            if not in_position:
                # Look for breakout entry
                if current_close > breakout_level and current_volume_surge >= volume_multiplier:
                    signals.iloc[i] = 1
                    in_position = True
                    entry_price = current_close
            else:
                # We're in a position, look for exit
                # Exit on breakdown below recent support
                if current_close < breakdown_level:
                    signals.iloc[i] = -1
                    in_position = False
                    entry_price = None
                # Exit on retracement from entry
                elif entry_price is not None:
                    retracement = ((entry_price - current_close) / entry_price) * 100
                    if retracement > retracement_threshold:
                        signals.iloc[i] = -1
                        in_position = False
                        entry_price = None
        
        return signals
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data meets strategy requirements."""
        # Check for required columns
        required_cols = ['Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                return False
        
        # Check for sufficient data
        required_periods = self.get_required_periods()
        if len(data) < required_periods:
            return False
        
        # Check for null values in required columns
        for col in required_cols:
            if data[col].isna().any():
                return False
        
        # Check for non-positive prices
        price_cols = ['Close', 'High', 'Low']
        for col in price_cols:
            if (data[col] <= 0).any():
                return False
        
        # Check for non-positive volume
        if (data['Volume'] <= 0).any():
            return False
        
        # Check price relationships (High >= Close >= Low)
        if not (data['High'] >= data['Close']).all():
            return False
        if not (data['Close'] >= data['Low']).all():
            return False
        
        return True
    
    def get_required_periods(self) -> int:
        """Get minimum number of periods required for signal generation."""
        lookback = self.get_parameter('lookback_period')
        # Need lookback period plus volume MA period (20) plus buffer
        return lookback + 25