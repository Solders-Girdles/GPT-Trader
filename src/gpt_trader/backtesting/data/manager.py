"""Historical data manager with caching."""

import json
from datetime import datetime
from pathlib import Path

from gpt_trader.backtesting.data.fetcher import CoinbaseHistoricalFetcher
from gpt_trader.features.brokerages.core.interfaces import Candle
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="historical_data")


class HistoricalDataManager:
    """
    Manage historical candle data with file-based caching.

    This manager:
    - Checks cache for requested data
    - Fetches missing data from API
    - Stores fetched data in cache
    - Provides efficient access to historical candles

    Future Enhancement:
    - Migrate from JSON to Parquet for better compression and query performance
    - Add DuckDB integration for SQL queries
    - Implement data quality checks
    """

    def __init__(
        self,
        fetcher: CoinbaseHistoricalFetcher,
        cache_dir: Path,
    ):
        """
        Initialize data manager.

        Args:
            fetcher: Historical data fetcher
            cache_dir: Directory for cache storage
        """
        self.fetcher = fetcher
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache metadata {symbol: {granularity: [(start, end)]}}
        self._coverage_index: dict[str, dict[str, list[tuple[datetime, datetime]]]] = {}
        self._load_coverage_index()

    async def get_candles(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """
        Get candles from cache or fetch from API.

        Args:
            symbol: Trading pair
            granularity: Candle granularity
            start: Start time (inclusive)
            end: End time (exclusive)

        Returns:
            List of candles sorted by timestamp
        """
        # Check cache
        cached_candles = self._read_from_cache(symbol, granularity, start, end)

        # Identify gaps
        gaps = self._identify_gaps(symbol, granularity, start, end)

        if not gaps:
            # All data in cache
            return cached_candles

        # Fetch missing data
        fetched_candles = []
        for gap_start, gap_end in gaps:
            gap_candles = await self.fetcher.fetch_candles(
                symbol=symbol,
                granularity=granularity,
                start=gap_start,
                end=gap_end,
            )
            fetched_candles.extend(gap_candles)

            # Update cache
            self._write_to_cache(symbol, granularity, gap_candles)
            self._update_coverage(symbol, granularity, gap_start, gap_end)

        # Combine and sort
        all_candles = cached_candles + fetched_candles
        all_candles = sorted(all_candles, key=lambda c: c.ts)

        # Filter to requested range
        all_candles = [c for c in all_candles if start <= c.ts < end]

        return all_candles

    def _get_cache_path(self, symbol: str, granularity: str) -> Path:
        """Get cache file path for symbol and granularity."""
        safe_symbol = symbol.replace("/", "_").replace("-", "_")
        return self.cache_dir / f"{safe_symbol}_{granularity}.json"

    def _read_from_cache(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[Candle]:
        """Read candles from cache file."""
        cache_path = self._get_cache_path(symbol, granularity)

        if not cache_path.exists():
            return []

        try:
            with open(cache_path) as f:
                data = json.load(f)

            candles = []
            for item in data.get("candles", []):
                ts = datetime.fromisoformat(item["ts"])
                if start <= ts < end:
                    candles.append(
                        Candle(
                            ts=ts,
                            open=float(item["open"]),
                            high=float(item["high"]),
                            low=float(item["low"]),
                            close=float(item["close"]),
                            volume=float(item["volume"]),
                        )
                    )

            return candles
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(
                "Failed to read candles from cache",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="read_from_cache",
                symbol=symbol,
                granularity=granularity,
                cache_path=str(cache_path),
            )
            return []

    def _write_to_cache(
        self,
        symbol: str,
        granularity: str,
        candles: list[Candle],
    ) -> None:
        """Write candles to cache file (append mode)."""
        cache_path = self._get_cache_path(symbol, granularity)

        # Read existing
        existing = []
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                    existing = data.get("candles", [])
            except json.JSONDecodeError as exc:
                logger.error(
                    "Failed to decode existing cache file",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="write_to_cache",
                    symbol=symbol,
                    granularity=granularity,
                    cache_path=str(cache_path),
                )
                existing = []

        # Convert new candles to dicts
        new_items = [
            {
                "ts": c.ts.isoformat(),
                "open": str(c.open),
                "high": str(c.high),
                "low": str(c.low),
                "close": str(c.close),
                "volume": str(c.volume),
            }
            for c in candles
        ]

        # Merge and deduplicate
        existing_timestamps = {item["ts"] for item in existing}
        for item in new_items:
            if item["ts"] not in existing_timestamps:
                existing.append(item)

        # Sort by timestamp
        existing.sort(key=lambda x: x["ts"])

        # Write back
        with open(cache_path, "w") as f:
            json.dump({"candles": existing}, f, indent=2)

    def _identify_gaps(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, datetime]]:
        """
        Identify time gaps not covered by cache.

        Args:
            symbol: Trading pair
            granularity: Candle granularity
            start: Requested start time
            end: Requested end time

        Returns:
            List of (gap_start, gap_end) tuples
        """
        if symbol not in self._coverage_index:
            # No coverage at all
            return [(start, end)]

        if granularity not in self._coverage_index[symbol]:
            # No coverage for this granularity
            return [(start, end)]

        covered_ranges = self._coverage_index[symbol][granularity]

        # Simple gap detection: if no coverage overlaps [start, end], it's a gap
        # This is simplified; a production version would do proper interval merging
        has_coverage = any(
            cov_start <= start < cov_end or cov_start < end <= cov_end
            for cov_start, cov_end in covered_ranges
        )

        if has_coverage:
            return []  # Assume complete coverage for simplicity
        else:
            return [(start, end)]

    def _update_coverage(
        self,
        symbol: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> None:
        """Update coverage index with newly cached data."""
        if symbol not in self._coverage_index:
            self._coverage_index[symbol] = {}

        if granularity not in self._coverage_index[symbol]:
            self._coverage_index[symbol][granularity] = []

        self._coverage_index[symbol][granularity].append((start, end))
        self._save_coverage_index()

    def _load_coverage_index(self) -> None:
        """Load coverage index from disk."""
        index_path = self.cache_dir / "_coverage_index.json"

        if not index_path.exists():
            return

        try:
            with open(index_path) as f:
                data = json.load(f)

            # Deserialize datetimes
            for symbol, granularities in data.items():
                self._coverage_index[symbol] = {}
                for granularity, ranges in granularities.items():
                    self._coverage_index[symbol][granularity] = [
                        (datetime.fromisoformat(r[0]), datetime.fromisoformat(r[1])) for r in ranges
                    ]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error(
                "Failed to load coverage index",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="load_coverage_index",
                index_path=str(index_path),
            )

    def _save_coverage_index(self) -> None:
        """Save coverage index to disk."""
        index_path = self.cache_dir / "_coverage_index.json"

        # Serialize datetimes
        data = {}
        for symbol, granularities in self._coverage_index.items():
            data[symbol] = {}
            for granularity, ranges in granularities.items():
                data[symbol][granularity] = [(r[0].isoformat(), r[1].isoformat()) for r in ranges]

        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)
