"""
Mean reversion strategy using RSI and statistical bands.
Exploits tendency of prices to revert to their statistical mean.
"""

import pandas as pd
import numpy as np
from .base import StrategyBase, StrategyConfig


class MeanReversionStrategy(StrategyBase):
    """
    Mean reversion strategy using RSI and standard deviation bands.
    
    Buys when prices are oversold and below statistical mean.
    Sells when prices are overbought and above statistical mean.
    
    Logic:
    - Calculate RSI for momentum assessment
    - Calculate mean and standard deviation bands
    - Buy when RSI < oversold threshold AND price < lower band
    - Sell when RSI > overbought threshold AND price > upper band
    """
    
    def __init__(
        self, 
        rsi_period: int = 14,
        mean_period: int = 20,
        std_multiplier: float = 1.5,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            rsi_period: Period for RSI calculation
            mean_period: Period for mean and standard deviation calculation
            std_multiplier: Multiplier for standard deviation bands
            oversold_threshold: RSI threshold for oversold condition
            overbought_threshold: RSI threshold for overbought condition
        """
        config = StrategyConfig(
            name="MeanReversion",
            description=f"RSI mean reversion strategy ({rsi_period}d RSI, "
                       f"{mean_period}d bands, {oversold_threshold}/{overbought_threshold} thresholds)",
            parameters={
                'rsi_period': rsi_period,
                'mean_period': mean_period,
                'std_multiplier': std_multiplier,
                'oversold_threshold': oversold_threshold,
                'overbought_threshold': overbought_threshold
            }
        )
        super().__init__(config)
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Handle division by zero
        avg_loss = avg_loss.replace(0, np.finfo(float).eps)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate mean reversion trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series of signals: 1 = buy, -1 = sell, 0 = hold
        """
        if 'Close' not in data.columns:
            raise ValueError("Data must have 'Close' column")
        
        # Get parameters
        rsi_period = self.get_parameter('rsi_period')
        mean_period = self.get_parameter('mean_period')
        std_multiplier = self.get_parameter('std_multiplier')
        oversold_threshold = self.get_parameter('oversold_threshold')
        overbought_threshold = self.get_parameter('overbought_threshold')
        
        # Calculate indicators
        prices = data['Close']
        rsi = self._calculate_rsi(prices, rsi_period)
        
        # Calculate mean and standard deviation bands
        rolling_mean = prices.rolling(window=mean_period).mean()
        rolling_std = prices.rolling(window=mean_period).std()
        
        upper_band = rolling_mean + (std_multiplier * rolling_std)
        lower_band = rolling_mean - (std_multiplier * rolling_std)
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Generate signals
        for i in range(max(rsi_period, mean_period), len(data)):
            current_price = prices.iloc[i]
            current_rsi = rsi.iloc[i]
            current_upper = upper_band.iloc[i]
            current_lower = lower_band.iloc[i]
            
            # Skip if we don't have valid data
            if any(pd.isna(x) for x in [current_price, current_rsi, current_upper, current_lower]):
                continue
            
            # Buy signal: Oversold RSI and price below lower band
            if current_rsi < oversold_threshold and current_price < current_lower:
                signals.iloc[i] = 1
                
            # Sell signal: Overbought RSI and price above upper band
            elif current_rsi > overbought_threshold and current_price > current_upper:
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
        
        # Check for non-positive prices
        if (data['Close'] <= 0).any():
            return False
        
        return True
    
    def get_required_periods(self) -> int:
        """Get minimum number of periods required for signal generation."""
        rsi_period = self.get_parameter('rsi_period')
        mean_period = self.get_parameter('mean_period')
        # Need the maximum of RSI and mean periods plus buffer
        return max(rsi_period, mean_period) + 5