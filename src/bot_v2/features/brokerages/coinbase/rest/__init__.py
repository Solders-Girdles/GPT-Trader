"""Modular mixins composing the Coinbase REST service."""

from bot_v2.features.brokerages.coinbase.rest.base import CoinbaseRestServiceBase, logger
from bot_v2.features.brokerages.coinbase.rest.orders import OrderRestMixin
from bot_v2.features.brokerages.coinbase.rest.pnl import PnLRestMixin
from bot_v2.features.brokerages.coinbase.rest.portfolio import PortfolioRestMixin
from bot_v2.features.brokerages.coinbase.rest.products import ProductRestMixin

__all__ = [
    "CoinbaseRestServiceBase",
    "OrderRestMixin",
    "PortfolioRestMixin",
    "ProductRestMixin",
    "PnLRestMixin",
    "logger",
]
