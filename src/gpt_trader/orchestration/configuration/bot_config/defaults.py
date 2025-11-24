"""Default symbol sets and risk configuration paths for BotConfig."""

from __future__ import annotations

from pathlib import Path

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

DEFAULT_SPOT_RISK_PATH = Path(__file__).resolve().parents[5] / "config" / "risk" / "spot_top10.json"

__all__ = [
    "TOP_VOLUME_BASES",
    "DEFAULT_SPOT_SYMBOLS",
    "DEFAULT_SPOT_RISK_PATH",
]
