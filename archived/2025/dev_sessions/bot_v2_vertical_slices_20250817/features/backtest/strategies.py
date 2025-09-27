"""
Local strategy implementations for backtesting.

This is a self-contained module with all strategies needed for backtesting.
No external dependencies - everything is local to maintain isolation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BacktestStrategy(ABC):
    """Base class for backtest strategies."""
    
    def __init__(self, **params):
        """Initialize with parameters."""
        self.params = params
    
    @abstractmethod
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals from data."""
        pass
    
    @abstractmethod
    def get_required_periods(self) -> int:
        """Return minimum periods needed."""
        pass


class SimpleMAStrategy(BacktestStrategy):
    """Simple moving average crossover strategy."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30, **kwargs):
        super().__init__(fast_period=fast_period, slow_period=slow_period, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate MA crossover signals."""
        fast_ma = data['close'].rolling(window=self.fast_period).mean()
        slow_ma = data['close'].rolling(window=self.slow_period).mean()
        
        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0
        
        # Buy when fast crosses above slow
        signals[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1
        # Sell when fast crosses below slow
        signals[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1
        
        return signals.fillna(0)
    
    def get_required_periods(self) -> int:
        """Need at least slow_period + 1 for crossover detection."""
        return self.slow_period + 1


class MomentumStrategy(BacktestStrategy):
    """Momentum-based trading strategy."""
    
    def __init__(self, lookback: int = 20, threshold: float = 0.02, **kwargs):
        super().__init__(lookback=lookback, threshold=threshold, **kwargs)
        self.lookback = lookback
        self.threshold = threshold
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals."""
        returns = data['close'].pct_change(self.lookback)
        
        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0
        
        # Buy on strong positive momentum
        signals[returns > self.threshold] = 1
        # Sell on strong negative momentum
        signals[returns < -self.threshold] = -1
        
        return signals.fillna(0)
    
    def get_required_periods(self) -> int:
        """Need lookback + 1 periods."""
        return self.lookback + 1


class MeanReversionStrategy(BacktestStrategy):
    """Mean reversion trading strategy."""
    
    def __init__(self, period: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__(period=period, num_std=num_std, **kwargs)
        self.period = period
        self.num_std = num_std
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals using Bollinger Bands."""
        mean = data['close'].rolling(window=self.period).mean()
        std = data['close'].rolling(window=self.period).std()
        
        upper_band = mean + (std * self.num_std)
        lower_band = mean - (std * self.num_std)
        
        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0
        
        # Buy when price touches lower band (oversold)
        signals[data['close'] <= lower_band] = 1
        # Sell when price touches upper band (overbought)
        signals[data['close'] >= upper_band] = -1
        
        return signals.fillna(0)
    
    def get_required_periods(self) -> int:
        """Need period periods for stats."""
        return self.period


class VolatilityStrategy(BacktestStrategy):
    """Volatility-based trading strategy."""
    
    def __init__(self, period: int = 20, vol_threshold: float = 0.02, **kwargs):
        super().__init__(period=period, vol_threshold=vol_threshold, **kwargs)
        self.period = period
        self.vol_threshold = vol_threshold
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on volatility."""
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=self.period).std()
        
        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0
        
        # Trade only in low volatility environments
        low_vol = volatility < self.vol_threshold
        
        # Use simple momentum in low vol periods
        momentum = data['close'].pct_change(self.period)
        signals[low_vol & (momentum > 0.01)] = 1
        signals[low_vol & (momentum < -0.01)] = -1
        
        return signals.fillna(0)
    
    def get_required_periods(self) -> int:
        """Need period + 1 for volatility calc."""
        return self.period + 1


class BreakoutStrategy(BacktestStrategy):
    """Price breakout trading strategy."""
    
    def __init__(self, lookback: int = 20, **kwargs):
        super().__init__(lookback=lookback, **kwargs)
        self.lookback = lookback
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Generate breakout signals."""
        high_rolling = data['high'].rolling(window=self.lookback).max()
        low_rolling = data['low'].rolling(window=self.lookback).min()
        
        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0
        
        # Buy on upward breakout
        signals[data['close'] > high_rolling.shift(1)] = 1
        # Sell on downward breakout
        signals[data['close'] < low_rolling.shift(1)] = -1
        
        return signals.fillna(0)
    
    def get_required_periods(self) -> int:
        """Need lookback + 1 periods."""
        return self.lookback + 1


# Local strategy factory - no external dependencies
STRATEGY_MAP = {
    'SimpleMAStrategy': SimpleMAStrategy,
    'MomentumStrategy': MomentumStrategy,
    'MeanReversionStrategy': MeanReversionStrategy,
    'VolatilityStrategy': VolatilityStrategy,
    'BreakoutStrategy': BreakoutStrategy,
}


def create_local_strategy(name: str, **params) -> BacktestStrategy:
    """
    Create a strategy instance locally.
    
    Args:
        name: Strategy name
        **params: Strategy parameters
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy not found
    """
    if name not in STRATEGY_MAP:
        available = ', '.join(STRATEGY_MAP.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    
    strategy_class = STRATEGY_MAP[name]
    return strategy_class(**params)


def list_local_strategies() -> list:
    """List available local strategies."""
    return list(STRATEGY_MAP.keys())