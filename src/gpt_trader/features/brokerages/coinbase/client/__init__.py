"""
Exports for Coinbase client.
"""

from gpt_trader.features.brokerages.coinbase.auth import (
    CoinbaseAuth,
    SimpleAuth,
    create_cdp_jwt_auth,
)
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient

__all__ = ["SimpleAuth", "CoinbaseAuth", "CoinbaseClient", "create_cdp_jwt_auth"]
