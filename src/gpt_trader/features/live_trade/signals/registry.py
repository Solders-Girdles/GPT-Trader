"""
Signal registry for dynamic signal instantiation.

Maps signal names to their classes and configuration classes,
enabling YAML-based configuration of ensemble strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.signals.protocol import SignalGenerator


@dataclass
class SignalRegistration:
    """Registration entry for a signal type."""

    signal_class: type
    config_class: type | None
    description: str = ""


# Global registry of available signals
_SIGNAL_REGISTRY: dict[str, SignalRegistration] = {}


def register_signal(
    name: str,
    signal_class: type,
    config_class: type | None = None,
    description: str = "",
) -> None:
    """Register a signal type for dynamic instantiation.

    Args:
        name: Unique identifier for the signal (e.g., "trend", "order_flow").
        signal_class: The SignalGenerator implementation class.
        config_class: Optional dataclass for configuration.
        description: Human-readable description.
    """
    _SIGNAL_REGISTRY[name] = SignalRegistration(
        signal_class=signal_class,
        config_class=config_class,
        description=description,
    )


def get_signal_registration(name: str) -> SignalRegistration | None:
    """Get registration for a signal by name."""
    return _SIGNAL_REGISTRY.get(name)


def list_registered_signals() -> list[str]:
    """List all registered signal names."""
    return list(_SIGNAL_REGISTRY.keys())


def create_signal(name: str, parameters: dict[str, Any] | None = None) -> SignalGenerator:
    """Create a signal instance from registry.

    Args:
        name: Registered signal name.
        parameters: Configuration parameters to pass to config class.

    Returns:
        Configured SignalGenerator instance.

    Raises:
        ValueError: If signal name is not registered.
    """
    registration = _SIGNAL_REGISTRY.get(name)
    if registration is None:
        available = ", ".join(_SIGNAL_REGISTRY.keys())
        raise ValueError(f"Unknown signal '{name}'. Available: {available}")

    if registration.config_class is not None and parameters:
        config = registration.config_class(**parameters)
        return registration.signal_class(config)
    elif registration.config_class is not None:
        return registration.signal_class(registration.config_class())
    else:
        return registration.signal_class()


def _register_builtin_signals() -> None:
    """Register all built-in signals."""
    # Import here to avoid circular imports
    from gpt_trader.features.live_trade.signals.mean_reversion import (
        MeanReversionSignal,
        MeanReversionSignalConfig,
    )
    from gpt_trader.features.live_trade.signals.momentum import (
        MomentumSignal,
        MomentumSignalConfig,
    )
    from gpt_trader.features.live_trade.signals.order_flow import (
        OrderFlowSignal,
        OrderFlowSignalConfig,
    )
    from gpt_trader.features.live_trade.signals.orderbook_imbalance import (
        OrderbookImbalanceSignal,
        OrderbookImbalanceSignalConfig,
    )
    from gpt_trader.features.live_trade.signals.spread import (
        SpreadSignal,
        SpreadSignalConfig,
    )
    from gpt_trader.features.live_trade.signals.trend import (
        TrendSignal,
        TrendSignalConfig,
    )
    from gpt_trader.features.live_trade.signals.vwap import (
        VWAPSignal,
        VWAPSignalConfig,
    )

    # Core signals
    register_signal(
        "trend",
        TrendSignal,
        TrendSignalConfig,
        "Trend-following signal using EMA crossover and ADX",
    )
    register_signal(
        "mean_reversion",
        MeanReversionSignal,
        MeanReversionSignalConfig,
        "Mean reversion signal using Bollinger Bands",
    )
    register_signal(
        "momentum",
        MomentumSignal,
        MomentumSignalConfig,
        "Momentum signal using RSI",
    )

    # Market microstructure signals
    register_signal(
        "order_flow",
        OrderFlowSignal,
        OrderFlowSignalConfig,
        "Order flow signal based on trade aggressor ratio",
    )
    register_signal(
        "orderbook_imbalance",
        OrderbookImbalanceSignal,
        OrderbookImbalanceSignalConfig,
        "Orderbook imbalance signal from bid/ask depth",
    )
    register_signal(
        "spread",
        SpreadSignal,
        SpreadSignalConfig,
        "Market quality signal based on bid-ask spread",
    )
    register_signal(
        "vwap",
        VWAPSignal,
        VWAPSignalConfig,
        "Mean reversion signal based on VWAP deviation",
    )


# Auto-register on module import
_register_builtin_signals()
