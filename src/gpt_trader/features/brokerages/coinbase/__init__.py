"""
Coinbase brokerage module.

Exports the modern CoinbaseClient and associated authentication helpers.
"""

from gpt_trader.features.brokerages.coinbase.client import CoinbaseAuth, CoinbaseClient, SimpleAuth

__all__ = ["CoinbaseClient", "CoinbaseAuth", "SimpleAuth"]
