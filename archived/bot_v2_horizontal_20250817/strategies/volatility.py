"""
Volatility-based trading strategy using ATR and price range analysis.
Trades based on volatility expansion and contraction patterns.
"""

import pandas as pd
import numpy as np
from .base import StrategyBase, StrategyConfig


class VolatilityStrategy(StrategyBase):
    """
    Volatility strategy using ATR and price range indicators.
    
    Buys during low volatility (consolidation) anticipating breakout.
    Sells during high volatility (expansion) to take profits or cut losses.
    
    Logic:
    - Calculate ATR for volatility measurement
    - Calculate volatility percentile over lookback period
    - Buy when volatility is in bottom percentile (quiet consolidation)
    - Sell when volatility is in top percentile (volatile expansion)
    """
    
    def __init__(
        self, 
        atr_period: int = 14,
        lookback_period: int = 50,
        low_vol_percentile: float = 25.0,
        high_vol_percentile: float = 75.0,
        volume_confirmation: bool = True
    ):
        """
        Initialize volatility strategy.
        
        Args:
            atr_period: Period for ATR calculation
            lookback_period: Period for volatility percentile calculation
            low_vol_percentile: Percentile threshold for low volatility (buy signal)
            high_vol_percentile: Percentile threshold for high volatility (sell signal)
            volume_confirmation: Whether to require volume confirmation
        """
        config = StrategyConfig(
            name="Volatility",
            description=f"ATR volatility strategy ({atr_period}d ATR, "
                       f"{low_vol_percentile}/{high_vol_percentile} percentiles)",
            parameters={
                'atr_period': atr_period,
                'lookback_period': lookback_period,
                'low_vol_percentile': low_vol_percentile,
                'high_vol_percentile': high_vol_percentile,
                'volume_confirmation': volume_confirmation
            }
        )
        super().__init__(config)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = data['High']
        low = data['Low']
        close = data['Close'].shift(1)
        
        # Calculate true range
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as exponential moving average of true range
        atr = true_range.rolling(window=period, min_periods=period).mean()
        
        return atr
    
    def _calculate_volatility_percentile(self, atr: pd.Series, lookback: int) -> pd.Series:
        """Calculate rolling percentile rank of volatility."""
        percentile = pd.Series(index=atr.index, dtype=float)
        
        for i in range(lookback, len(atr)):
            window = atr.iloc[i-lookback+1:i+1]
            current = atr.iloc[i]
            
            if pd.notna(current) and len(window.dropna()) >= lookback // 2:
                # Calculate percentile rank
                rank = (window < current).sum() / len(window.dropna()) * 100
                percentile.iloc[i] = rank
        
        return percentile
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate volatility-based trading signals.
        
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
        atr_period = self.get_parameter('atr_period')
        lookback_period = self.get_parameter('lookback_period')
        low_vol_percentile = self.get_parameter('low_vol_percentile')
        high_vol_percentile = self.get_parameter('high_vol_percentile')
        volume_confirmation = self.get_parameter('volume_confirmation')
        
        # Calculate indicators
        atr = self._calculate_atr(data, atr_period)
        vol_percentile = self._calculate_volatility_percentile(atr, lookback_period)
        
        # Calculate volume indicators if needed
        if volume_confirmation:
            volume_ma = data['Volume'].rolling(window=20).mean()
            relative_volume = data['Volume'] / volume_ma
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Generate signals
        min_period = max(atr_period, lookback_period) + 1
        
        for i in range(min_period, len(data)):
            current_percentile = vol_percentile.iloc[i]
            
            # Skip if we don't have valid data
            if pd.isna(current_percentile):
                continue
            
            # Volume confirmation check
            volume_ok = True
            if volume_confirmation:
                current_rel_vol = relative_volume.iloc[i] if i < len(relative_volume) else np.nan
                volume_ok = not pd.isna(current_rel_vol)
            
            # Buy signal: Low volatility (consolidation phase)
            # This indicates potential for a breakout
            if current_percentile <= low_vol_percentile and volume_ok:
                signals.iloc[i] = 1
                
            # Sell signal: High volatility (expansion phase)
            # This indicates potential exhaustion or need for risk management
            elif current_percentile >= high_vol_percentile:
                signals.iloc[i] = -1
        
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
        
        return True
    
    def get_required_periods(self) -> int:
        """Get minimum number of periods required for signal generation."""
        atr_period = self.get_parameter('atr_period')
        lookback_period = self.get_parameter('lookback_period')
        # Need lookback period plus ATR period plus buffer
        return max(atr_period, lookback_period) + 5