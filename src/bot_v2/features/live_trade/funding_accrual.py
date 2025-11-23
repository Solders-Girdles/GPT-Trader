"""Enhanced funding accrual tracking with hourly TWAP-based buckets.

This module extends the PnL tracking with:
- Hourly funding accrual buckets based on TWAP
- Mid-day and end-of-day settlement events
- Detailed funding history for analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from bot_v2.persistence.event_store import EventStore

logger = get_logger(__name__, component="funding_accrual")


@dataclass
class FundingAccrualBucket:
    """Hourly bucket for funding accrual tracking."""

    symbol: str
    hour_start: datetime
    hour_end: datetime
    position_size: Decimal
    side: str | None  # 'long' or 'short'
    funding_rate: Decimal
    mark_price_twap: Decimal  # Time-weighted average mark price for the hour
    mark_price_samples: list[tuple[datetime, Decimal]] = field(default_factory=list)
    accrued_funding: Decimal = Decimal("0")
    is_settled: bool = False


@dataclass
class FundingSettlementEvent:
    """Record of a funding settlement event."""

    event_type: str  # "MIDDAY", "EOD", or "FUNDING_PAYMENT"
    timestamp: datetime
    symbol: str
    total_funding_accrued: Decimal
    buckets_settled: list[FundingAccrualBucket]
    position_snapshot: dict[str, Any]


class FundingAccrualTracker:
    """Tracks funding accrual with hourly TWAP-based buckets."""

    # Settlement times (UTC)
    MIDDAY_SETTLEMENT_TIME = time(12, 0, 0)  # 12:00 UTC
    EOD_SETTLEMENT_TIME = time(0, 0, 0)  # 00:00 UTC (next day)

    def __init__(self, event_store: EventStore | None = None, bot_id: str = "unknown") -> None:
        """Initialize funding accrual tracker.

        Args:
            event_store: Optional event store for recording settlement events
            bot_id: Bot identifier for event logging
        """
        self._event_store = event_store
        self._bot_id = bot_id

        # Current accrual buckets by symbol
        self._current_buckets: dict[str, FundingAccrualBucket] = {}

        # Historical buckets for analysis
        self._historical_buckets: dict[str, list[FundingAccrualBucket]] = {}

        # Settlement history
        self._settlement_history: list[FundingSettlementEvent] = []

        # Last settlement times
        self._last_midday_settlement: datetime | None = None
        self._last_eod_settlement: datetime | None = None

    def update_mark_price_sample(
        self,
        symbol: str,
        mark_price: Decimal,
        position_size: Decimal,
        side: str | None,
        funding_rate: Decimal,
        timestamp: datetime | None = None,
    ) -> None:
        """Add a mark price sample to the current hour's bucket.

        Args:
            symbol: The perpetual symbol
            mark_price: Current mark price
            position_size: Current position size
            side: Position side ('long' or 'short')
            funding_rate: Current funding rate
            timestamp: Sample timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Get or create current bucket for this hour
        bucket = self._get_or_create_bucket(symbol, timestamp, position_size, side, funding_rate)

        # Add mark price sample
        bucket.mark_price_samples.append((timestamp, mark_price))

        # Update TWAP
        bucket.mark_price_twap = self._calculate_twap(bucket)

        # Accrue funding based on TWAP
        self._accrue_funding_for_bucket(bucket, position_size, side, funding_rate)

    def _get_or_create_bucket(
        self,
        symbol: str,
        timestamp: datetime,
        position_size: Decimal,
        side: str | None,
        funding_rate: Decimal,
    ) -> FundingAccrualBucket:
        """Get or create accrual bucket for the current hour."""
        hour_start = timestamp.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)

        # Check if we need a new bucket (new hour)
        current_bucket = self._current_buckets.get(symbol)
        if current_bucket is None or current_bucket.hour_end <= timestamp:
            # Settle old bucket if it exists
            if current_bucket is not None:
                self._settle_bucket(current_bucket)

            # Create new bucket
            current_bucket = FundingAccrualBucket(
                symbol=symbol,
                hour_start=hour_start,
                hour_end=hour_end,
                position_size=position_size,
                side=side,
                funding_rate=funding_rate,
                mark_price_twap=Decimal("0"),
            )
            self._current_buckets[symbol] = current_bucket

        return current_bucket

    def _calculate_twap(self, bucket: FundingAccrualBucket) -> Decimal:
        """Calculate time-weighted average price from samples.

        Uses simple average for now; could be enhanced with actual time weighting.
        """
        if not bucket.mark_price_samples:
            return Decimal("0")

        total = sum(price for _, price in bucket.mark_price_samples)
        return total / Decimal(len(bucket.mark_price_samples))

    def _accrue_funding_for_bucket(
        self,
        bucket: FundingAccrualBucket,
        position_size: Decimal,
        side: str | None,
        funding_rate: Decimal,
    ) -> None:
        """Accrue funding for a bucket based on TWAP.

        Funding accrues continuously throughout the hour based on:
        - Position size
        - TWAP mark price
        - Funding rate (typically hourly rate)
        """
        if position_size == Decimal("0") or side is None:
            return

        # Calculate hourly funding based on TWAP
        notional = position_size * bucket.mark_price_twap
        hourly_funding = notional * funding_rate

        # Adjust sign based on side
        if side == "long":
            bucket.accrued_funding = -hourly_funding  # Longs pay
        else:
            bucket.accrued_funding = hourly_funding  # Shorts receive

    def _settle_bucket(self, bucket: FundingAccrualBucket) -> None:
        """Settle a completed bucket and move to history."""
        if bucket.is_settled:
            return

        bucket.is_settled = True

        # Move to historical buckets
        if bucket.symbol not in self._historical_buckets:
            self._historical_buckets[bucket.symbol] = []
        self._historical_buckets[bucket.symbol].append(bucket)

        logger.debug(
            "Funding bucket settled",
            operation="funding_accrual",
            stage="settle_bucket",
            symbol=bucket.symbol,
            hour_start=bucket.hour_start.isoformat(),
            accrued_funding=float(bucket.accrued_funding),
        )

    def check_and_execute_settlements(
        self, current_time: datetime | None = None
    ) -> list[FundingSettlementEvent]:
        """Check for and execute mid-day or EOD settlements.

        Args:
            current_time: Current time (defaults to now)

        Returns:
            List of settlement events that occurred
        """
        if current_time is None:
            current_time = datetime.now(UTC)

        events: list[FundingSettlementEvent] = []

        # Check for mid-day settlement (12:00 UTC)
        if self._should_execute_settlement(
            current_time, self.MIDDAY_SETTLEMENT_TIME, self._last_midday_settlement
        ):
            event = self._execute_settlement("MIDDAY", current_time)
            if event:
                events.append(event)
                self._last_midday_settlement = current_time

        # Check for EOD settlement (00:00 UTC next day)
        if self._should_execute_settlement(
            current_time, self.EOD_SETTLEMENT_TIME, self._last_eod_settlement
        ):
            event = self._execute_settlement("EOD", current_time)
            if event:
                events.append(event)
                self._last_eod_settlement = current_time

        return events

    def _should_execute_settlement(
        self,
        current_time: datetime,
        settlement_time: time,
        last_settlement: datetime | None,
    ) -> bool:
        """Check if a settlement should be executed."""
        # Get settlement datetime for today
        settlement_datetime = datetime.combine(current_time.date(), settlement_time, tzinfo=UTC)

        # If we haven't settled today and current time is past settlement time
        if last_settlement is None:
            return current_time >= settlement_datetime

        # Check if we need a new settlement (past settlement time and haven't settled today)
        return current_time >= settlement_datetime and last_settlement.date() < current_time.date()

    def _execute_settlement(
        self, event_type: str, timestamp: datetime
    ) -> FundingSettlementEvent | None:
        """Execute a funding settlement and record to event store.

        Args:
            event_type: "MIDDAY" or "EOD"
            timestamp: Settlement timestamp

        Returns:
            FundingSettlementEvent or None if no positions to settle
        """
        logger.info(
            "Executing funding settlement",
            operation="funding_accrual",
            stage="settlement",
            event_type=event_type,
            timestamp=timestamp.isoformat(),
        )

        # Collect all buckets to settle (all symbols)
        all_buckets: list[FundingAccrualBucket] = []
        total_funding_by_symbol: dict[str, Decimal] = {}

        for symbol, bucket in self._current_buckets.items():
            # Settle current bucket
            self._settle_bucket(bucket)
            all_buckets.append(bucket)

            # Calculate total funding for symbol
            symbol_history = self._historical_buckets.get(symbol, [])
            total_funding = sum(b.accrued_funding for b in symbol_history)
            total_funding_by_symbol[symbol] = total_funding

        if not all_buckets:
            logger.debug(
                "No buckets to settle",
                operation="funding_accrual",
                stage="settlement",
                event_type=event_type,
            )
            return None

        # Create settlement event (using first symbol for simplicity)
        # In practice, you might want one event per symbol
        symbol = all_buckets[0].symbol if all_buckets else "unknown"
        event = FundingSettlementEvent(
            event_type=event_type,
            timestamp=timestamp,
            symbol=symbol,
            total_funding_accrued=total_funding_by_symbol.get(symbol, Decimal("0")),
            buckets_settled=all_buckets,
            position_snapshot={
                "symbols": list(total_funding_by_symbol.keys()),
                "total_funding_by_symbol": {
                    s: float(f) for s, f in total_funding_by_symbol.items()
                },
            },
        )

        # Record to event store
        if self._event_store is not None:
            try:
                self._event_store.append_metric(
                    bot_id=self._bot_id,
                    metric_type=f"funding_settlement_{event_type.lower()}",
                    data={
                        "event_type": event_type,
                        "timestamp": timestamp.isoformat(),
                        "symbol": symbol,
                        "total_funding": str(event.total_funding_accrued),
                        "buckets_count": len(all_buckets),
                        "symbols": list(total_funding_by_symbol.keys()),
                        "funding_by_symbol": {
                            s: str(f) for s, f in total_funding_by_symbol.items()
                        },
                    },
                )
            except Exception as exc:
                logger.error(
                    "Failed to record funding settlement to event store",
                    operation="funding_accrual",
                    stage="settlement",
                    error=str(exc),
                    exc_info=True,
                )

        # Add to settlement history
        self._settlement_history.append(event)

        logger.info(
            "Funding settlement complete",
            operation="funding_accrual",
            stage="settlement_complete",
            event_type=event_type,
            symbols_count=len(total_funding_by_symbol),
            total_funding=float(sum(total_funding_by_symbol.values())),
        )

        return event

    def get_current_accrued_funding(self, symbol: str) -> Decimal:
        """Get currently accrued funding for a symbol."""
        bucket = self._current_buckets.get(symbol)
        if bucket is None:
            return Decimal("0")
        return bucket.accrued_funding

    def get_total_historical_funding(self, symbol: str) -> Decimal:
        """Get total historical funding for a symbol."""
        history = self._historical_buckets.get(symbol, [])
        return sum(b.accrued_funding for b in history)

    def get_settlement_history(self, event_type: str | None = None) -> list[FundingSettlementEvent]:
        """Get settlement history, optionally filtered by event type."""
        if event_type is None:
            return list(self._settlement_history)
        return [e for e in self._settlement_history if e.event_type == event_type]


__all__ = [
    "FundingAccrualBucket",
    "FundingAccrualTracker",
    "FundingSettlementEvent",
]
