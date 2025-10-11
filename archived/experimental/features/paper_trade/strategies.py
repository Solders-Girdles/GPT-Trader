"""
Local strategy implementations for paper trading.

Complete duplication from backtest - intentional for isolation!
No external dependencies.
"""

from abc import ABC, abstractmethod

import pandas as pd


class PaperTradeStrategy(ABC):
    """Base class for paper trading strategies."""

    def __init__(self, **params) -> None:
        """Initialize with parameters."""
        self.params = params
        self.position = 0  # Track current position

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> int:
        """
        Analyze data and return signal.
        Returns: 1 (buy), -1 (sell), 0 (hold)
        """

    @abstractmethod
    def get_required_periods(self) -> int:
        """Return minimum periods needed."""


class SimpleMAStrategy(PaperTradeStrategy):
    """Simple moving average crossover strategy."""

    def __init__(self, fast_period: int = 10, slow_period: int = 30, **kwargs) -> None:
        super().__init__(fast_period=fast_period, slow_period=slow_period, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def analyze(self, data: pd.DataFrame) -> int:
        """Generate MA crossover signal for latest data."""
        if len(data) < self.slow_period:
            return 0

        fast_ma = data["close"].rolling(window=self.fast_period).mean()
        slow_ma = data["close"].rolling(window=self.slow_period).mean()

        # Get current and previous values
        curr_fast = fast_ma.iloc[-1]
        curr_slow = slow_ma.iloc[-1]
        prev_fast = fast_ma.iloc[-2]
        prev_slow = slow_ma.iloc[-2]

        # Detect crossover
        if curr_fast > curr_slow and prev_fast <= prev_slow:
            return 1  # Buy signal
        elif curr_fast < curr_slow and prev_fast >= prev_slow:
            return -1  # Sell signal

        return 0  # Hold

    def get_required_periods(self) -> int:
        return self.slow_period + 1


class MomentumStrategy(PaperTradeStrategy):
    """Momentum-based trading strategy."""

    def __init__(self, lookback: int = 20, threshold: float = 0.02, **kwargs) -> None:
        super().__init__(lookback=lookback, threshold=threshold, **kwargs)
        self.lookback = lookback
        self.threshold = threshold

    def analyze(self, data: pd.DataFrame) -> int:
        """Generate momentum signal for latest data."""
        if len(data) < self.lookback + 1:
            return 0

        # Calculate momentum
        current_price = data["close"].iloc[-1]
        lookback_price = data["close"].iloc[-self.lookback - 1]
        momentum = (current_price - lookback_price) / lookback_price

        # Generate signal
        if momentum > self.threshold:
            return 1  # Strong positive momentum
        elif momentum < -self.threshold:
            return -1  # Strong negative momentum

        return 0  # No signal

    def get_required_periods(self) -> int:
        return self.lookback + 1


class MeanReversionStrategy(PaperTradeStrategy):
    """Mean reversion trading strategy."""

    def __init__(self, period: int = 20, num_std: float = 2.0, **kwargs) -> None:
        super().__init__(period=period, num_std=num_std, **kwargs)
        self.period = period
        self.num_std = num_std

    def analyze(self, data: pd.DataFrame) -> int:
        """Generate mean reversion signal using Bollinger Bands."""
        if len(data) < self.period:
            return 0

        close = data["close"].iloc[-self.period :]
        mean = close.mean()
        std = close.std()

        upper_band = mean + (std * self.num_std)
        lower_band = mean - (std * self.num_std)
        current_price = close.iloc[-1]

        # Generate signal
        if current_price <= lower_band:
            return 1  # Oversold - buy
        elif current_price >= upper_band:
            return -1  # Overbought - sell

        return 0  # Within bands

    def get_required_periods(self) -> int:
        return self.period


class VolatilityStrategy(PaperTradeStrategy):
    """Volatility-based trading strategy."""

    def __init__(self, period: int = 20, vol_threshold: float = 0.02, **kwargs) -> None:
        super().__init__(period=period, vol_threshold=vol_threshold, **kwargs)
        self.period = period
        self.vol_threshold = vol_threshold

    def analyze(self, data: pd.DataFrame) -> int:
        """Generate signals based on volatility."""
        if len(data) < self.period + 1:
            return 0

        returns = data["close"].pct_change()
        current_vol = returns.iloc[-self.period :].std()

        # Only trade in low volatility
        if current_vol >= self.vol_threshold:
            return 0  # Too volatile

        # Use simple momentum in low vol
        momentum = (data["close"].iloc[-1] - data["close"].iloc[-self.period]) / data["close"].iloc[
            -self.period
        ]

        if momentum > 0.01:
            return 1
        elif momentum < -0.01:
            return -1

        return 0

    def get_required_periods(self) -> int:
        return self.period + 1


class BreakoutStrategy(PaperTradeStrategy):
    """Price breakout trading strategy."""

    def __init__(self, lookback: int = 20, **kwargs) -> None:
        super().__init__(lookback=lookback, **kwargs)
        self.lookback = lookback

    def analyze(self, data: pd.DataFrame) -> int:
        """Generate breakout signal."""
        if len(data) < self.lookback + 1:
            return 0

        # Get recent high/low
        recent_high = data["high"].iloc[-self.lookback - 1 : -1].max()
        recent_low = data["low"].iloc[-self.lookback - 1 : -1].min()
        current_price = data["close"].iloc[-1]

        # Check for breakout
        if current_price > recent_high:
            return 1  # Upward breakout
        elif current_price < recent_low:
            return -1  # Downward breakout

        return 0  # No breakout

    def get_required_periods(self) -> int:
        return self.lookback + 1


# Local strategy factory
STRATEGY_MAP = {
    "SimpleMAStrategy": SimpleMAStrategy,
    "MomentumStrategy": MomentumStrategy,
    "MeanReversionStrategy": MeanReversionStrategy,
    "VolatilityStrategy": VolatilityStrategy,
    "BreakoutStrategy": BreakoutStrategy,
}


def create_paper_strategy(name: str, **params) -> PaperTradeStrategy:
    """
    Create a strategy instance for paper trading.

    Args:
        name: Strategy name
        **params: Strategy parameters

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy not found
    """
    if name not in STRATEGY_MAP:
        available = ", ".join(STRATEGY_MAP.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")

    strategy_class = STRATEGY_MAP[name]
    return strategy_class(**params)
