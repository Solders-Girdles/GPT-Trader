"""Historical data fetcher for Coinbase API."""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from bot_v2.features.brokerages.coinbase.client.client import CoinbaseClient
from bot_v2.features.brokerages.core.interfaces import Candle


class CoinbaseHistoricalFetcher:
    """
    Fetch historical candle data from Coinbase Advanced Trade API.

    Reference: https://docs.cloud.coinbase.com/advanced-trade-api/reference/retailbrokerageapi_getcandles

    Endpoint: GET /api/v3/brokerage/products/{product_id}/candles
    Rate Limit: 10 requests/second (public endpoints)
    Max Candles: 300 per request
    """

    def __init__(
        self,
        client: CoinbaseClient,
        rate_limit_rps: int = 10,
    ):
        """
        Initialize fetcher.

        Args:
            client: Coinbase API client
            rate_limit_rps: Rate limit (requests per second)
        """
        self.client = client
        self.rate_limit_rps = rate_limit_rps
        self._rate_limit_delay = 1.0 / rate_limit_rps
        self._last_request_time = 0.0

    async def fetch_candles(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """
        Fetch candles for a symbol in a time range.

        This method automatically chunks large date ranges into 300-candle
        batches and handles rate limiting.

        Args:
            symbol: Trading pair (e.g., "BTC-PERP-USDC")
            granularity: Candle granularity (e.g., "ONE_MINUTE", "FIVE_MINUTE")
            start: Start of time range (inclusive)
            end: End of time range (exclusive)

        Returns:
            List of candles sorted by timestamp (ascending)
        """
        # Calculate chunk size based on granularity
        candle_duration = self._granularity_to_seconds(granularity)
        max_candles_per_request = 300

        # Split into chunks
        chunks = self._create_chunks(start, end, candle_duration, max_candles_per_request)

        # Fetch chunks with rate limiting
        all_candles = []
        for chunk_start, chunk_end in chunks:
            candles = await self._fetch_chunk(
                symbol=symbol,
                granularity=granularity,
                start=chunk_start,
                end=chunk_end,
            )
            all_candles.extend(candles)

            # Rate limit
            await self._rate_limit()

        # Sort and deduplicate
        all_candles = self._deduplicate_candles(all_candles)

        return all_candles

    async def _fetch_chunk(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """
        Fetch a single chunk of candles from API.

        Args:
            symbol: Trading pair
            granularity: Candle granularity
            start: Chunk start time
            end: Chunk end time

        Returns:
            List of candles for this chunk
        """
        # Convert to Unix timestamps
        start_unix = int(start.timestamp())
        end_unix = int(end.timestamp())

        # Build request
        params = {
            "start": str(start_unix),
            "end": str(end_unix),
            "granularity": granularity,
        }

        # Call API
        response = await self.client.get(
            f"/api/v3/brokerage/products/{symbol}/candles",
            params=params,
        )

        # Parse response
        candles = []
        if "candles" in response:
            for candle_data in response["candles"]:
                candles.append(self._parse_candle(candle_data))

        return candles

    def _parse_candle(self, data: dict) -> Candle:
        """Parse API response into Candle object."""
        return Candle(
            ts=datetime.fromtimestamp(int(data["start"])),
            open=Decimal(data["open"]),
            high=Decimal(data["high"]),
            low=Decimal(data["low"]),
            close=Decimal(data["close"]),
            volume=Decimal(data["volume"]),
        )

    def _granularity_to_seconds(self, granularity: str) -> int:
        """Convert granularity string to seconds."""
        mapping = {
            "ONE_MINUTE": 60,
            "FIVE_MINUTE": 300,
            "FIFTEEN_MINUTE": 900,
            "THIRTY_MINUTE": 1800,
            "ONE_HOUR": 3600,
            "TWO_HOUR": 7200,
            "SIX_HOUR": 21600,
            "ONE_DAY": 86400,
        }
        return mapping.get(granularity, 60)

    def _create_chunks(
        self,
        start: datetime,
        end: datetime,
        candle_seconds: int,
        max_candles: int,
    ) -> list[tuple[datetime, datetime]]:
        """
        Split date range into chunks of max_candles.

        Args:
            start: Start date
            end: End date
            candle_seconds: Duration of each candle in seconds
            max_candles: Maximum candles per chunk

        Returns:
            List of (chunk_start, chunk_end) tuples
        """
        chunks = []
        chunk_duration_seconds = candle_seconds * max_candles
        current_start = start

        while current_start < end:
            chunk_end = current_start + timedelta(seconds=chunk_duration_seconds)
            if chunk_end > end:
                chunk_end = end

            chunks.append((current_start, chunk_end))
            current_start = chunk_end

        return chunks

    def _deduplicate_candles(self, candles: list[Candle]) -> list[Candle]:
        """Remove duplicate candles and sort by timestamp."""
        seen_timestamps = set()
        unique_candles = []

        for candle in sorted(candles, key=lambda c: c.ts):
            if candle.ts not in seen_timestamps:
                unique_candles.append(candle)
                seen_timestamps.add(candle.ts)

        return unique_candles

    async def _rate_limit(self) -> None:
        """Apply rate limiting delay."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - time_since_last)

        self._last_request_time = asyncio.get_event_loop().time()
