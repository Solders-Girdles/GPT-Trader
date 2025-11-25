"""
Exports for Coinbase client.
"""

from gpt_trader.features.brokerages.coinbase.auth import CoinbaseAuth, SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient

__all__ = ["SimpleAuth", "CoinbaseAuth", "CoinbaseClient"]
