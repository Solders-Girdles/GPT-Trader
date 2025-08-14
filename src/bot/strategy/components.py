"""
Component-Based Strategy Building Framework.
Provides reusable, tested components for building trading strategies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd
from bot.indicators.enhanced import (
    bollinger_bands,
    enhanced_atr,
    enhanced_rsi,
    regime_filter,
    time_based_filters,
    volatility_regime,
    volume_breakout,
)
from bot.strategy.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class ComponentConfig:
    """Configuration for a strategy component."""

    component_type: str
    parameters: dict[str, Any]
    enabled: bool = True
    priority: int = 1  # Higher priority components are applied first


class BaseComponent(ABC):
    """Base class for all strategy components."""

    def __init__(self, config: ComponentConfig) -> None:
        self.config = config
        self.component_type = config.component_type
        self.parameters = config.parameters
        self.enabled = config.enabled
        self.priority = config.priority

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.Series:
        """Process market data and return component signals."""
        pass

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        """Get parameter bounds for optimization."""
        return {}

    def validate_parameters(self) -> bool:
        """Validate component parameters."""
        return True


class EntryComponent(BaseComponent):
    """Base class for entry signal components."""

    def __init__(self, config: ComponentConfig) -> None:
        super().__init__(config)
        self.component_type = "entry"


class ExitComponent(BaseComponent):
    """Base class for exit signal components."""

    def __init__(self, config: ComponentConfig) -> None:
        super().__init__(config)
        self.component_type = "exit"


class RiskComponent(BaseComponent):
    """Base class for risk management components."""

    def __init__(self, config: ComponentConfig) -> None:
        super().__init__(config)
        self.component_type = "risk"


class FilterComponent(BaseComponent):
    """Base class for filter components."""

    def __init__(self, config: ComponentConfig) -> None:
        super().__init__(config)
        self.component_type = "filter"


# Entry Components


class DonchianBreakoutEntry(EntryComponent):
    """Donchian channel breakout entry component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate entry signals based on Donchian channel breakouts."""
        lookback = self.parameters.get("lookback", 55)
        atr_period = self.parameters.get("atr_period", 20)
        atr_k = self.parameters.get("atr_k", 2.0)

        # Calculate Donchian channels
        high_channel = data["High"].rolling(window=lookback).max()
        low_channel = data["Low"].rolling(window=lookback).min()

        # Calculate ATR
        atr = enhanced_atr(data, atr_period)

        # Entry signals
        long_signal = (data["Close"] > high_channel - atr_k * atr) & (
            data["Close"].shift(1) <= high_channel.shift(1) - atr_k * atr.shift(1)
        )
        short_signal = (data["Close"] < low_channel + atr_k * atr) & (
            data["Close"].shift(1) >= low_channel.shift(1) + atr_k * atr.shift(1)
        )

        signals = pd.Series(0, index=data.index)
        signals[long_signal] = 1
        signals[short_signal] = -1

        return signals

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {
            "lookback": {"min": 10, "max": 200, "type": "int"},
            "atr_period": {"min": 5, "max": 50, "type": "int"},
            "atr_k": {"min": 0.5, "max": 5.0, "type": "float"},
        }


class RSIEntry(EntryComponent):
    """RSI-based entry component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate entry signals based on RSI levels."""
        period = self.parameters.get("period", 14)
        oversold = self.parameters.get("oversold", 30.0)
        overbought = self.parameters.get("overbought", 70.0)

        # Calculate RSI
        rsi = enhanced_rsi(data, period)

        # Entry signals
        long_signal = (rsi < oversold) & (rsi.shift(1) >= oversold)
        short_signal = (rsi > overbought) & (rsi.shift(1) <= overbought)

        signals = pd.Series(0, index=data.index)
        signals[long_signal] = 1
        signals[short_signal] = -1

        return signals

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {
            "period": {"min": 5, "max": 30, "type": "int"},
            "oversold": {"min": 20.0, "max": 40.0, "type": "float"},
            "overbought": {"min": 60.0, "max": 80.0, "type": "float"},
        }


class VolumeBreakoutEntry(EntryComponent):
    """Volume-based breakout entry component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate entry signals based on volume breakouts."""
        period = self.parameters.get("period", 20)
        threshold = self.parameters.get("threshold", 1.5)

        # Calculate volume breakout
        volume_signal = volume_breakout(data, period, threshold)

        # Combine with price movement
        price_change = data["Close"].pct_change()
        long_signal = volume_signal & (price_change > 0.01)  # 1% price increase
        short_signal = volume_signal & (price_change < -0.01)  # 1% price decrease

        signals = pd.Series(0, index=data.index)
        signals[long_signal] = 1
        signals[short_signal] = -1

        return signals

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {
            "period": {"min": 10, "max": 50, "type": "int"},
            "threshold": {"min": 1.0, "max": 3.0, "type": "float"},
        }


# Exit Components


class FixedTargetExit(ExitComponent):
    """Fixed target exit component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate exit signals based on fixed profit/loss targets."""
        profit_target = self.parameters.get("profit_target", 0.05)  # 5%
        stop_loss = self.parameters.get("stop_loss", 0.03)  # 3%

        # This would typically work with position tracking
        # For now, return a simple signal based on price movement
        price_change = data["Close"].pct_change()

        signals = pd.Series(0, index=data.index)
        signals[price_change > profit_target] = -1  # Exit long
        signals[price_change < -profit_target] = 1  # Exit short
        signals[price_change < -stop_loss] = -1  # Stop loss for long
        signals[price_change > stop_loss] = 1  # Stop loss for short

        return signals

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {
            "profit_target": {"min": 0.01, "max": 0.20, "type": "float"},
            "stop_loss": {"min": 0.01, "max": 0.10, "type": "float"},
        }


class TrailingStopExit(ExitComponent):
    """Trailing stop exit component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate exit signals based on trailing stops."""
        atr_period = self.parameters.get("atr_period", 20)
        self.parameters.get("atr_multiplier", 2.0)

        # Calculate ATR
        enhanced_atr(data, atr_period)

        # Simple trailing stop logic
        # In practice, this would track positions and trailing stops
        signals = pd.Series(0, index=data.index)

        return signals

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {
            "atr_period": {"min": 5, "max": 50, "type": "int"},
            "atr_multiplier": {"min": 1.0, "max": 4.0, "type": "float"},
        }


class TimeBasedExit(ExitComponent):
    """Time-based exit component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate exit signals based on time."""
        self.parameters.get("max_hold_days", 30)

        # Simple time-based exit
        # In practice, this would track position entry dates
        signals = pd.Series(0, index=data.index)

        return signals

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {"max_hold_days": {"min": 1, "max": 100, "type": "int"}}


# Risk Components


class PositionSizingRisk(RiskComponent):
    """Position sizing risk management component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Calculate position sizes based on risk parameters."""
        risk_per_trade = self.parameters.get("risk_per_trade", 0.02)  # 2%
        method = self.parameters.get("method", "atr")

        if method == "atr":
            atr_period = self.parameters.get("atr_period", 20)
            atr = enhanced_atr(data, atr_period)
            # Position size based on ATR
            position_size = risk_per_trade / (atr * 2)  # 2 ATR stop loss
        else:
            # Fixed position size
            position_size = pd.Series(risk_per_trade, index=data.index)

        return position_size

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {
            "risk_per_trade": {"min": 0.01, "max": 0.05, "type": "float"},
            "method": {"type": "categorical", "values": ["atr", "fixed", "kelly"]},
            "atr_period": {"min": 5, "max": 50, "type": "int"},
        }


class CorrelationFilterRisk(RiskComponent):
    """Correlation-based risk filter component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate risk signals based on correlation."""
        self.parameters.get("threshold", 0.7)
        self.parameters.get("lookback", 60)

        # Simple correlation filter
        # In practice, this would compare with market index
        signals = pd.Series(1, index=data.index)  # Allow trading by default

        return signals

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {
            "threshold": {"min": 0.5, "max": 0.9, "type": "float"},
            "lookback": {"min": 30, "max": 100, "type": "int"},
        }


# Filter Components


class RegimeFilter(FilterComponent):
    """Market regime filter component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate filter signals based on market regime."""
        lookback = self.parameters.get("lookback", 200)

        # Apply regime filter
        regime_ok = regime_filter(data, lookback)

        return regime_ok

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {"lookback": {"min": 100, "max": 300, "type": "int"}}


class VolatilityFilter(FilterComponent):
    """Volatility-based filter component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate filter signals based on volatility."""
        short_period = self.parameters.get("short_period", 20)
        long_period = self.parameters.get("long_period", 100)
        threshold = self.parameters.get("threshold", 1.2)

        # Calculate volatility regime
        vol_regime = volatility_regime(data, short_period, long_period)

        # Filter based on volatility
        signals = vol_regime <= threshold  # Allow trading in normal/low volatility

        return signals

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {
            "short_period": {"min": 10, "max": 50, "type": "int"},
            "long_period": {"min": 50, "max": 200, "type": "int"},
            "threshold": {"min": 1.0, "max": 2.0, "type": "float"},
        }


class BollingerFilter(FilterComponent):
    """Bollinger Bands filter component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate filter signals based on Bollinger Bands."""
        period = self.parameters.get("period", 20)
        std_dev = self.parameters.get("std_dev", 2.0)

        # Calculate Bollinger Bands
        upper, middle, lower = bollinger_bands(data, period, std_dev)

        # Filter based on price position relative to bands
        signals = (data["Close"] > lower) & (data["Close"] < upper)

        return signals

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {
            "period": {"min": 10, "max": 50, "type": "int"},
            "std_dev": {"min": 1.5, "max": 3.0, "type": "float"},
        }


class TimeFilter(FilterComponent):
    """Time-based filter component."""

    def process(self, data: pd.DataFrame) -> pd.Series:
        """Generate filter signals based on time."""
        day_of_week = self.parameters.get("day_of_week")
        month = self.parameters.get("month")

        # Apply time-based filters
        signals = time_based_filters(data, day_of_week, month)

        return signals

    def get_parameter_bounds(self) -> dict[str, dict[str, Any]]:
        return {
            "day_of_week": {"type": "categorical", "values": [None, 0, 1, 2, 3, 4]},
            "month": {"type": "categorical", "values": [None] + list(range(1, 13))},
        }


# Component Registry


class ComponentRegistry:
    """Registry for all available strategy components."""

    def __init__(self) -> None:
        self.components = {
            # Entry components
            "donchian_breakout": DonchianBreakoutEntry,
            "rsi_entry": RSIEntry,
            "volume_breakout": VolumeBreakoutEntry,
            # Exit components
            "fixed_target": FixedTargetExit,
            "trailing_stop": TrailingStopExit,
            "time_based": TimeBasedExit,
            # Risk components
            "position_sizing": PositionSizingRisk,
            "correlation_filter": CorrelationFilterRisk,
            # Filter components
            "regime_filter": RegimeFilter,
            "volatility_filter": VolatilityFilter,
            "bollinger_filter": BollingerFilter,
            "time_filter": TimeFilter,
        }

    def get_component(self, name: str, config: ComponentConfig) -> BaseComponent:
        """Get a component instance by name."""
        if name not in self.components:
            raise ValueError(f"Unknown component: {name}")

        component_class = self.components[name]
        return component_class(config)

    def list_components(self, component_type: str | None = None) -> list[str]:
        """List available components, optionally filtered by type."""
        if component_type is None:
            return list(self.components.keys())

        return [
            name
            for name, cls in self.components.items()
            if cls.__bases__[0].__name__ == f"{component_type.capitalize()}Component"
        ]

    def get_component_info(self, name: str) -> dict[str, Any]:
        """Get information about a component."""
        if name not in self.components:
            raise ValueError(f"Unknown component: {name}")

        component_class = self.components[name]
        instance = component_class(ComponentConfig(name, {}))

        return {
            "name": name,
            "class": component_class.__name__,
            "type": instance.component_type,
            "parameter_bounds": instance.get_parameter_bounds(),
            "description": component_class.__doc__ or "No description available",
        }


# Component-based Strategy Builder


class ComponentBasedStrategy(Strategy):
    """Strategy built from reusable components."""

    def __init__(self, components: list[BaseComponent]) -> None:
        super().__init__()
        self.components = sorted(components, key=lambda x: x.priority, reverse=True)
        self.registry = ComponentRegistry()

        # Separate components by type
        self.entry_components = [c for c in self.components if isinstance(c, EntryComponent)]
        self.exit_components = [c for c in self.components if isinstance(c, ExitComponent)]
        self.risk_components = [c for c in self.components if isinstance(c, RiskComponent)]
        self.filter_components = [c for c in self.components if isinstance(c, FilterComponent)]

    def generate_signals(
        self, bars: pd.DataFrame, market_data: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Generate trading signals using component pipeline."""
        df = bars.copy()
        df["signal"] = 0

        # Apply filters first
        filter_signal = pd.Series(True, index=df.index)
        for filter_comp in self.filter_components:
            if filter_comp.enabled:
                filter_signal = filter_signal & filter_comp.process(df)

        # Generate entry signals
        entry_signal = pd.Series(0, index=df.index)
        for entry_comp in self.entry_components:
            if entry_comp.enabled:
                component_signal = entry_comp.process(df)
                # Combine signals (take the first non-zero signal)
                entry_signal = entry_signal.where(entry_signal != 0, component_signal)

        # Generate exit signals
        exit_signal = pd.Series(0, index=df.index)
        for exit_comp in self.exit_components:
            if exit_comp.enabled:
                component_signal = exit_comp.process(df)
                exit_signal = exit_signal | component_signal  # Any exit signal triggers exit

        # Combine signals
        final_signal = entry_signal.copy()
        final_signal[exit_signal != 0] = exit_signal[exit_signal != 0]

        # Apply filters
        final_signal = final_signal.where(filter_signal, 0)

        df["signal"] = final_signal

        # Add component signals for debugging
        for i, comp in enumerate(self.components):
            if comp.enabled:
                df[f"signal_{comp.component_type}_{i}"] = comp.process(df)

        return df

    def get_position_size(self, bars: pd.DataFrame) -> pd.Series:
        """Calculate position sizes using risk components."""
        position_size = pd.Series(1.0, index=bars.index)  # Default size

        for risk_comp in self.risk_components:
            if risk_comp.enabled and isinstance(risk_comp, PositionSizingRisk):
                component_size = risk_comp.process(bars)
                position_size = position_size * component_size

        return position_size

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ComponentBasedStrategy:
        """Create a component-based strategy from configuration."""
        registry = ComponentRegistry()
        components = []

        for component_config in config.get("components", []):
            name = component_config["name"]
            params = component_config.get("parameters", {})
            enabled = component_config.get("enabled", True)
            priority = component_config.get("priority", 1)

            comp_config = ComponentConfig(name, params, enabled, priority)
            component = registry.get_component(name, comp_config)
            components.append(component)

        return cls(components)

    def to_config(self) -> dict[str, Any]:
        """Convert strategy to configuration."""
        config = {"strategy_type": "component_based", "components": []}

        for component in self.components:
            comp_config = {
                "name": component.__class__.__name__.lower(),
                "parameters": component.parameters,
                "enabled": component.enabled,
                "priority": component.priority,
            }
            config["components"].append(comp_config)

        return config
