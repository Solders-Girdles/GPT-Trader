"""
Unified Coinbase Client.
Combines all mixins into a single simple client class.
"""

from gpt_trader.features.brokerages.coinbase.client.accounts import AccountClientMixin
from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.client.market import MarketDataClientMixin
from gpt_trader.features.brokerages.coinbase.client.orders import OrderClientMixin
from gpt_trader.features.brokerages.coinbase.client.portfolio import PortfolioClientMixin
from gpt_trader.features.brokerages.coinbase.client.websocket_mixin import (
    WebSocketClientMixin,
)


class CoinbaseClient(
    CoinbaseClientBase,
    MarketDataClientMixin,
    OrderClientMixin,
    AccountClientMixin,
    PortfolioClientMixin,
    WebSocketClientMixin,
):
    """
    The unified Coinbase Client.
    Inherits base HTTP machinery and specific endpoint mixins.
    Includes WebSocket streaming for real-time market data.
    """

    pass


__all__ = ["CoinbaseClient"]
