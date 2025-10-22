"""Market data interface for backtesting and live trading."""

from datetime import datetime
from decimal import Decimal
from typing import Protocol

from bot_v2.features.brokerages.core.interfaces import Candle


class IMarketData(Protocol):
    """
    Market data interface for both live and simulated environments.

    This interface abstracts market data access, allowing the same trading
    logic to work with both live data (from Coinbase WebSocket/REST) and
    historical data (from cached Parquet files).
    """

    def candles(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """
        Fetch candles for the given symbol and time range.

        Args:
            symbol: Trading pair (e.g., "BTC-PERP-USDC")
            granularity: Candle granularity (e.g., "ONE_MINUTE", "FIVE_MINUTE")
            start: Start of time range (inclusive)
            end: End of time range (exclusive)

        Returns:
            List of candles sorted by timestamp (ascending)

        Raises:
            DataNotAvailableError: If data is not available for the requested range
        """
        ...

    def best_bid_ask(self, symbol: str) -> tuple[Decimal, Decimal]:
        """
        Get current best bid and ask quotes.

        Args:
            symbol: Trading pair

        Returns:
            Tuple of (best_bid, best_ask) prices

        Raises:
            QuoteNotAvailableError: If quote is not available
        """
        ...

    def mark_price(self, symbol: str) -> Decimal:
        """
        Get current mark price (typically mid of best bid/ask).

        Args:
            symbol: Trading pair

        Returns:
            Current mark price

        Raises:
            QuoteNotAvailableError: If quote is not available
        """
        ...

    def spread(self, symbol: str) -> Decimal:
        """
        Get current bid/ask spread.

        Args:
            symbol: Trading pair

        Returns:
            Spread in quote currency (ask - bid)
        """
        ...

    def spread_bps(self, symbol: str) -> Decimal:
        """
        Get current spread in basis points.

        Args:
            symbol: Trading pair

        Returns:
            Spread in basis points ((ask - bid) / mid * 10000)
        """
        ...
