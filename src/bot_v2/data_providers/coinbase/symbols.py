"""Symbol normalization helpers for the Coinbase data provider."""

from __future__ import annotations

from bot_v2.orchestration.runtime_settings import RuntimeSettings
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_provider")


class CoinbaseSymbolMixin:
    """Shared symbol helpers for Coinbase market data."""

    _settings: RuntimeSettings

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol format for Coinbase.

        Converts equity-style symbols (AAPL, BTC) to Coinbase format (BTC-USD by
        default, BTC-PERP when derivatives are enabled).
        """
        if "-" in symbol:
            return symbol.upper()

        symbol_upper = symbol.upper()
        derivatives_enabled = self._settings.coinbase_enable_derivatives

        spot_map = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "SOL": "SOL-USD",
            "DOGE": "DOGE-USD",
            "AVAX": "AVAX-USD",
            "LINK": "LINK-USD",
            "ADA": "ADA-USD",
            "DOT": "DOT-USD",
            "MATIC": "MATIC-USD",
            "UNI": "UNI-USD",
        }

        perp_map = {
            "BTC": "BTC-PERP",
            "ETH": "ETH-PERP",
            "SOL": "SOL-PERP",
            "DOGE": "DOGE-PERP",
            "AVAX": "AVAX-PERP",
            "LINK": "LINK-PERP",
            "ADA": "ADA-PERP",
            "DOT": "DOT-PERP",
            "MATIC": "MATIC-PERP",
            "UNI": "UNI-PERP",
        }

        default_quote = self._settings.coinbase_default_quote
        if default_quote not in {"USD", "USDC", "USDT"}:
            logger.debug(
                "Invalid default quote %s; falling back to USD",
                default_quote,
                operation="symbol_normalization",
            )
            default_quote = "USD"

        if derivatives_enabled and symbol_upper in perp_map:
            return perp_map[symbol_upper]

        if symbol_upper in spot_map:
            base, _ = spot_map[symbol_upper].split("-")
            return f"{base}-{default_quote}"

        return f"{symbol_upper}-{default_quote}"


__all__ = ["CoinbaseSymbolMixin"]
