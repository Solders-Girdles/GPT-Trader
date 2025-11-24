"""
Coinbase brokerage module.

Exports the modern CoinbaseClient and associated authentication helpers.
"""

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient, CoinbaseAuth, SimpleAuth

__all__ = ["CoinbaseClient", "CoinbaseAuth", "SimpleAuth"]