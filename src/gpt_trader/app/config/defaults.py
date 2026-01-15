"""Default symbol sets for BotConfig."""

from __future__ import annotations

# Top spot markets we enable by default (ordered by Coinbase USD volume).
TOP_VOLUME_BASES = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "LTC",
    "ADA",
    "DOGE",
    "BCH",
    "AVAX",
    "LINK",
]

DEFAULT_SPOT_SYMBOLS = [f"{base}-USD" for base in TOP_VOLUME_BASES]

__all__ = [
    "TOP_VOLUME_BASES",
    "DEFAULT_SPOT_SYMBOLS",
]
