"""
Signal generation and combination components for the Ensemble Strategy.

Includes:
- Market microstructure signals (OrderFlow, OrderbookImbalance, Spread, VWAP)
- Signal registry for dynamic instantiation from configuration
"""

from gpt_trader.features.live_trade.signals.order_flow import (
    OrderFlowSignal,
    OrderFlowSignalConfig,
)
from gpt_trader.features.live_trade.signals.orderbook_imbalance import (
    OrderbookImbalanceSignal,
    OrderbookImbalanceSignalConfig,
)
from gpt_trader.features.live_trade.signals.registry import (
    create_signal,
    get_signal_registration,
    list_registered_signals,
    register_signal,
)
from gpt_trader.features.live_trade.signals.spread import (
    SpreadSignal,
    SpreadSignalConfig,
)
from gpt_trader.features.live_trade.signals.vwap import (
    VWAPSignal,
    VWAPSignalConfig,
)

__all__ = [
    # Signals
    "OrderFlowSignal",
    "OrderFlowSignalConfig",
    "OrderbookImbalanceSignal",
    "OrderbookImbalanceSignalConfig",
    "SpreadSignal",
    "SpreadSignalConfig",
    "VWAPSignal",
    "VWAPSignalConfig",
    # Registry
    "create_signal",
    "get_signal_registration",
    "list_registered_signals",
    "register_signal",
]
