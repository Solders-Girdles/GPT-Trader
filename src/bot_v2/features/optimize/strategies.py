"""
Local strategy implementations for optimization.

Complete duplication - intentional for isolation!
"""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class OptimizableStrategy(ABC):
    """Base class for optimizable strategies."""

    def __init__(self, **params) -> None:
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals."""
        pass

    @abstractmethod
    def get_required_periods(self) -> int:
        """Return minimum periods needed."""
        pass


class SimpleMAStrategy(OptimizableStrategy):
    """Moving average crossover strategy."""

    def __init__(self, fast_period: int = 10, slow_period: int = 30, **kwargs) -> None:
        super().__init__(fast_period=fast_period, slow_period=slow_period, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate MA crossover signals."""
        fast_ma = data["close"].rolling(window=self.fast_period).mean()
        slow_ma = data["close"].rolling(window=self.slow_period).mean()

        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0

        # Crossover signals
        signals[(fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))] = 1
        signals[(fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))] = -1

        return signals.fillna(0)

    def get_required_periods(self) -> int:
        return self.slow_period + 1


class MomentumStrategy(OptimizableStrategy):
    """Momentum-based strategy."""

    def __init__(
        self, lookback: int = 20, threshold: float = 0.02, hold_period: int = 5, **kwargs
    ) -> None:
        super().__init__(lookback=lookback, threshold=threshold, hold_period=hold_period, **kwargs)
        self.lookback = lookback
        self.threshold = threshold
        self.hold_period = hold_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum signals."""
        returns = data["close"].pct_change(self.lookback)

        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0

        # Entry signals
        signals[returns > self.threshold] = 1
        signals[returns < -self.threshold] = -1

        # Hold for specified period
        final_signals = pd.Series(index=data.index, dtype=int)
        final_signals[:] = 0

        position = 0
        hold_counter = 0

        for i in range(len(signals)):
            if signals.iloc[i] != 0 and position == 0:
                # New signal
                position = signals.iloc[i]
                hold_counter = self.hold_period
                final_signals.iloc[i] = position
            elif hold_counter > 0:
                # Holding period
                hold_counter -= 1
                if hold_counter == 0:
                    # Exit signal
                    final_signals.iloc[i] = -position
                    position = 0

        return final_signals

    def get_required_periods(self) -> int:
        return self.lookback + 1


class MeanReversionStrategy(OptimizableStrategy):
    """Mean reversion strategy."""

    def __init__(
        self, period: int = 20, entry_std: float = 2.0, exit_std: float = 0.5, **kwargs
    ) -> None:
        super().__init__(period=period, entry_std=entry_std, exit_std=exit_std, **kwargs)
        self.period = period
        self.entry_std = entry_std
        self.exit_std = exit_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals."""
        close = data["close"]
        mean = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()

        upper_entry = mean + (std * self.entry_std)
        lower_entry = mean - (std * self.entry_std)
        upper_exit = mean + (std * self.exit_std)
        lower_exit = mean - (std * self.exit_std)

        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0

        position = 0

        for i in range(self.period, len(close)):
            if position == 0:
                # Look for entry
                if close.iloc[i] <= lower_entry.iloc[i]:
                    signals.iloc[i] = 1  # Buy oversold
                    position = 1
                elif close.iloc[i] >= upper_entry.iloc[i]:
                    signals.iloc[i] = -1  # Sell overbought
                    position = -1
            elif position == 1:
                # Long position - look for exit
                if close.iloc[i] >= upper_exit.iloc[i]:
                    signals.iloc[i] = -1  # Exit long
                    position = 0
            elif position == -1:
                # Short position - look for exit
                if close.iloc[i] <= lower_exit.iloc[i]:
                    signals.iloc[i] = 1  # Exit short
                    position = 0

        return signals

    def get_required_periods(self) -> int:
        return self.period


class VolatilityStrategy(OptimizableStrategy):
    """Volatility-based strategy."""

    def __init__(
        self, vol_period: int = 20, vol_threshold: float = 0.02, trend_period: int = 50, **kwargs
    ) -> None:
        super().__init__(
            vol_period=vol_period, vol_threshold=vol_threshold, trend_period=trend_period, **kwargs
        )
        self.vol_period = vol_period
        self.vol_threshold = vol_threshold
        self.trend_period = trend_period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate volatility-filtered signals."""
        returns = data["close"].pct_change()
        volatility = returns.rolling(window=self.vol_period).std()
        trend = data["close"].rolling(window=self.trend_period).mean()

        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0

        # Only trade in low volatility with clear trend
        low_vol = volatility < self.vol_threshold
        uptrend = data["close"] > trend
        downtrend = data["close"] < trend

        # Entry signals
        prev_uptrend = uptrend.shift(1).fillna(False)
        prev_downtrend = downtrend.shift(1).fillna(False)
        signals[low_vol & uptrend & ~prev_uptrend] = 1
        signals[low_vol & downtrend & ~prev_downtrend] = -1

        # Exit on high volatility
        signals[~low_vol & (signals.shift(1) != 0)] = -signals.shift(1)

        return signals.fillna(0)

    def get_required_periods(self) -> int:
        return max(self.vol_period, self.trend_period) + 1


class BreakoutStrategy(OptimizableStrategy):
    """Price breakout strategy."""

    def __init__(
        self, lookback: int = 20, confirm_bars: int = 2, stop_loss: float = 0.02, **kwargs
    ) -> None:
        super().__init__(
            lookback=lookback, confirm_bars=confirm_bars, stop_loss=stop_loss, **kwargs
        )
        self.lookback = lookback
        self.confirm_bars = confirm_bars
        self.stop_loss = stop_loss

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate breakout signals."""
        high_rolling = data["high"].rolling(window=self.lookback).max()
        low_rolling = data["low"].rolling(window=self.lookback).min()

        signals = pd.Series(index=data.index, dtype=int)
        signals[:] = 0

        position = 0
        entry_price = 0
        confirm_count = 0

        for i in range(self.lookback + 1, len(data)):
            price = data["close"].iloc[i]

            if position == 0:
                # Check for breakout
                if price > high_rolling.iloc[i - 1]:
                    confirm_count += 1
                    if confirm_count >= self.confirm_bars:
                        signals.iloc[i] = 1
                        position = 1
                        entry_price = price
                        confirm_count = 0
                elif price < low_rolling.iloc[i - 1]:
                    confirm_count += 1
                    if confirm_count >= self.confirm_bars:
                        signals.iloc[i] = -1
                        position = -1
                        entry_price = price
                        confirm_count = 0
                else:
                    confirm_count = 0

            elif position == 1:
                # Check stop loss
                if price < entry_price * (1 - self.stop_loss):
                    signals.iloc[i] = -1
                    position = 0
                    entry_price = 0

            elif position == -1:
                # Check stop loss
                if price > entry_price * (1 + self.stop_loss):
                    signals.iloc[i] = 1
                    position = 0
                    entry_price = 0

        return signals

    def get_required_periods(self) -> int:
        return self.lookback + self.confirm_bars


# Strategy factory
STRATEGY_MAP = {
    "SimpleMA": SimpleMAStrategy,
    "Momentum": MomentumStrategy,
    "MeanReversion": MeanReversionStrategy,
    "Volatility": VolatilityStrategy,
    "Breakout": BreakoutStrategy,
}


def create_local_strategy(name: str, **params) -> OptimizableStrategy:
    """Create strategy instance for optimization."""
    if name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {name}")

    strategy_class = STRATEGY_MAP[name]
    return strategy_class(**params)


def get_strategy_params(name: str) -> dict[str, list[Any]]:
    """Get default parameter grid for optimization."""
    if name == "SimpleMA":
        return {"fast_period": [5, 10, 15, 20], "slow_period": [20, 30, 40, 50]}
    elif name == "Momentum":
        return {
            "lookback": [10, 20, 30],
            "threshold": [0.01, 0.02, 0.03],
            "hold_period": [3, 5, 10],
        }
    elif name == "MeanReversion":
        return {"period": [15, 20, 25], "entry_std": [1.5, 2.0, 2.5], "exit_std": [0.25, 0.5, 0.75]}
    elif name == "Volatility":
        return {
            "vol_period": [15, 20, 25],
            "vol_threshold": [0.015, 0.02, 0.025],
            "trend_period": [40, 50, 60],
        }
    elif name == "Breakout":
        return {
            "lookback": [15, 20, 25],
            "confirm_bars": [1, 2, 3],
            "stop_loss": [0.01, 0.02, 0.03],
        }
    else:
        return {}


def validate_params(name: str, params: dict[str, Any]) -> bool:
    """Validate strategy parameters."""
    # Add validation logic if needed
    return True
