"""Neutral entry point for the Coinbase Trader runtime.

The implementation lives in :mod:`bot_v2.orchestration.perps_bot`. This module provides a
spot-first alias while keeping the legacy ``PerpsBot`` name importable for backwards compatibility.
"""

from __future__ import annotations

from bot_v2.orchestration.perps_bot import CoinbaseTrader, PerpsBot

__all__ = ["CoinbaseTrader", "PerpsBot"]
