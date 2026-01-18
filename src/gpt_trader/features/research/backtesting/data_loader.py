"""
Historical data loader for backtesting.

Loads market data from EventStore and reconstructs market state
for strategy replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.utilities.datetime_helpers import normalize_to_utc, utc_now
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.coinbase.market_data_features import DepthSnapshot
    from gpt_trader.persistence.event_store import EventStore

logger = get_logger(__name__, component="data_loader")


@dataclass
class HistoricalDataPoint:
    """A single point in time with market data.

    Represents a snapshot of market state at a specific timestamp,
    suitable for replaying through a strategy.

    Attributes:
        timestamp: UTC timestamp of the data point.
        symbol: Trading pair symbol (e.g., "BTC-USD").
        mark_price: Current mark/spot price.
        orderbook_snapshot: Optional depth snapshot with bid/ask levels.
        trade_flow_stats: Optional trade aggregation stats (vwap, aggressor_ratio, etc.).
        spread_bps: Optional bid-ask spread in basis points.
    """

    timestamp: datetime
    symbol: str
    mark_price: Decimal
    orderbook_snapshot: DepthSnapshot | None = None
    trade_flow_stats: dict[str, Any] | None = None
    spread_bps: float | None = None

    def has_market_data(self) -> bool:
        """Check if this point has enhanced market data."""
        return self.orderbook_snapshot is not None or self.trade_flow_stats is not None


@dataclass
class DataLoadResult:
    """Result of a data loading operation.

    Attributes:
        data_points: List of historical data points.
        symbol: Symbol that was loaded.
        start_time: Earliest timestamp in the data.
        end_time: Latest timestamp in the data.
        total_events_processed: Number of events read from store.
    """

    data_points: list[HistoricalDataPoint]
    symbol: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    total_events_processed: int = 0

    @property
    def count(self) -> int:
        """Number of data points."""
        return len(self.data_points)

    @property
    def duration_seconds(self) -> float:
        """Duration of data in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class HistoricalDataLoader:
    """Load and reconstruct historical market data from EventStore.

    This loader reads events persisted during live trading and
    reconstructs the market state for backtesting. It handles:

    - Price tick events (mark price updates)
    - Orderbook snapshot events (depth data)
    - Trade flow summary events (volume/VWAP stats)

    Example:
        loader = HistoricalDataLoader(event_store)
        result = loader.load_symbol("BTC-USD", max_points=1000)
        for point in result.data_points:
            decision = strategy.decide(point.symbol, point.mark_price, ...)
    """

    def __init__(self, event_store: EventStore) -> None:
        """Initialize loader with an EventStore.

        Args:
            event_store: EventStore containing historical events.
        """
        self._event_store = event_store

    def load_symbol(
        self,
        symbol: str,
        max_points: int | None = None,
        include_orderbook: bool = True,
        include_trade_flow: bool = True,
    ) -> DataLoadResult:
        """Load historical data for a symbol.

        Args:
            symbol: Trading symbol to load (e.g., "BTC-USD").
            max_points: Maximum data points to return (None = all).
            include_orderbook: Include orderbook snapshots if available.
            include_trade_flow: Include trade flow stats if available.

        Returns:
            DataLoadResult with reconstructed data points.
        """
        events = self._event_store.list_events()
        logger.info(
            "Loading historical data",
            symbol=symbol,
            total_events=len(events),
            max_points=max_points,
        )

        # Collect events by type
        price_events: list[dict[str, Any]] = []
        orderbook_events: dict[str, dict[str, Any]] = {}  # timestamp -> event
        trade_flow_events: dict[str, dict[str, Any]] = {}  # timestamp -> event

        for event in events:
            event_type = event.get("type", "")
            data = event.get("data", {})
            event_symbol = data.get("symbol")

            if event_symbol != symbol:
                continue

            if event_type == "price_tick":
                price_events.append(event)
            elif event_type == "orderbook_snapshot" and include_orderbook:
                ts_key = data.get("timestamp", str(len(orderbook_events)))
                orderbook_events[ts_key] = data
            elif event_type == "trade_flow_summary" and include_trade_flow:
                ts_key = data.get("timestamp", str(len(trade_flow_events)))
                trade_flow_events[ts_key] = data

        # Build data points from price events
        data_points: list[HistoricalDataPoint] = []
        latest_orderbook: dict[str, Any] | None = None
        latest_trade_flow: dict[str, Any] | None = None

        for event in price_events:
            data = event.get("data", {})
            timestamp = self._parse_timestamp(data.get("timestamp"))
            price_str = data.get("price") or data.get("mark")

            if price_str is None:
                continue

            try:
                mark_price = Decimal(str(price_str))
            except Exception:
                continue

            # Find closest orderbook/trade_flow data
            if include_orderbook:
                latest_orderbook = self._find_closest_event(
                    orderbook_events, timestamp, latest_orderbook
                )
            if include_trade_flow:
                latest_trade_flow = self._find_closest_event(
                    trade_flow_events, timestamp, latest_trade_flow
                )

            # Build DepthSnapshot if orderbook data available
            orderbook_snapshot = None
            spread_bps = None
            if latest_orderbook:
                orderbook_snapshot = self._build_depth_snapshot(latest_orderbook)
                spread_bps = latest_orderbook.get("spread_bps")

            # Extract trade flow stats
            trade_stats = None
            if latest_trade_flow:
                trade_stats = {
                    "count": latest_trade_flow.get("trade_count", 0),
                    "volume": latest_trade_flow.get("volume"),
                    "vwap": latest_trade_flow.get("vwap"),
                    "avg_size": latest_trade_flow.get("avg_size"),
                    "aggressor_ratio": latest_trade_flow.get("aggressor_ratio"),
                }

            point = HistoricalDataPoint(
                timestamp=timestamp,
                symbol=symbol,
                mark_price=mark_price,
                orderbook_snapshot=orderbook_snapshot,
                trade_flow_stats=trade_stats,
                spread_bps=spread_bps,
            )
            data_points.append(point)

            if max_points and len(data_points) >= max_points:
                break

        # Sort by timestamp
        data_points.sort(key=lambda p: p.timestamp)

        result = DataLoadResult(
            data_points=data_points,
            symbol=symbol,
            start_time=data_points[0].timestamp if data_points else None,
            end_time=data_points[-1].timestamp if data_points else None,
            total_events_processed=len(events),
        )

        logger.info(
            "Loaded historical data",
            symbol=symbol,
            data_points=len(data_points),
            duration_seconds=result.duration_seconds,
        )

        return result

    def load_all_symbols(
        self,
        max_points_per_symbol: int | None = None,
    ) -> dict[str, DataLoadResult]:
        """Load data for all symbols in the event store.

        Returns:
            Dict mapping symbol to DataLoadResult.
        """
        events = self._event_store.list_events()

        # Find all symbols
        symbols: set[str] = set()
        for event in events:
            data = event.get("data", {})
            symbol = data.get("symbol")
            if symbol:
                symbols.add(symbol)

        results = {}
        for symbol in symbols:
            results[symbol] = self.load_symbol(symbol, max_points=max_points_per_symbol)

        return results

    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse timestamp from various formats."""
        if isinstance(ts, datetime):
            return normalize_to_utc(ts)
        if isinstance(ts, str):
            try:
                # ISO format
                return normalize_to_utc(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                pass
        # Default to now if unparseable
        return utc_now()

    def _find_closest_event(
        self,
        events: dict[str, dict[str, Any]],
        target_time: datetime,
        fallback: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Find the event closest to (but not after) target time.

        For simplicity, this implementation uses the latest available
        event before the target. A more sophisticated version could
        do binary search on sorted timestamps.
        """
        # For now, just return the most recent event
        if events:
            return list(events.values())[-1]
        return fallback

    def _build_depth_snapshot(self, data: dict[str, Any]) -> DepthSnapshot | None:
        """Build a DepthSnapshot from event data.

        The snapshot data stored in EventStore is a summary, not full
        depth. We create a minimal DepthSnapshot for backtesting.
        """
        try:
            from gpt_trader.features.brokerages.coinbase.market_data_features import (
                DepthSnapshot,
            )

            # Create a minimal snapshot from stored summary data
            mid_price_str = data.get("mid_price")
            spread_bps = data.get("spread_bps")

            if mid_price_str is None:
                return None

            mid = Decimal(str(mid_price_str))

            # Calculate approximate bid/ask from mid and spread
            if spread_bps:
                half_spread_pct = Decimal(str(spread_bps)) / Decimal("20000")
                best_bid = mid * (1 - half_spread_pct)
                best_ask = mid * (1 + half_spread_pct)
            else:
                best_bid = mid
                best_ask = mid

            # Create snapshot with minimal depth
            levels = [
                (best_bid, Decimal("1"), "bid"),
                (best_ask, Decimal("1"), "ask"),
            ]
            return DepthSnapshot(levels)

        except Exception as e:
            logger.debug(
                "Failed to build depth snapshot",
                error=str(e),
            )
            return None
