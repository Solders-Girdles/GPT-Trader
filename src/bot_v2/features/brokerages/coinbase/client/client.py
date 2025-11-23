"""
Unified Coinbase Client.
Combines all mixins into a single simple client class.
"""
from bot_v2.features.brokerages.coinbase.client.base import CoinbaseClientBase
from bot_v2.features.brokerages.coinbase.client.market import MarketDataClientMixin
from bot_v2.features.brokerages.coinbase.client.orders import OrderClientMixin
from bot_v2.features.brokerages.coinbase.client.accounts import AccountClientMixin
from bot_v2.features.brokerages.coinbase.client.portfolio import PortfolioClientMixin

class CoinbaseClient(
    CoinbaseClientBase,
    MarketDataClientMixin,
    OrderClientMixin,
    AccountClientMixin,
    PortfolioClientMixin
):
    """
    The unified Coinbase Client.
    Inherits base HTTP machinery and specific endpoint mixins.
    """
    pass

__all__ = ["CoinbaseClient"]
