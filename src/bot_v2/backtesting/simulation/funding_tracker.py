"""Funding PnL tracking for perpetual futures contracts."""

from datetime import datetime, timedelta
from decimal import Decimal


class FundingPnLTracker:
    """
    Track funding payments for perpetual futures positions.

    Funding rates are exchanged periodically (typically every 8 hours).
    On Coinbase, funding occurs at 00:00, 08:00, and 16:00 UTC.

    Funding payment = position_size * mark_price * funding_rate

    If holding a long position and funding rate is positive, you pay funding.
    If holding a short position and funding rate is positive, you receive funding.

    Reference: https://help.coinbase.com/en/coinbase/trading-and-funding/perpetual-futures/funding-rates
    """

    def __init__(
        self,
        accrual_interval_hours: int = 1,
        settlement_interval_hours: int = 8,
    ):
        """
        Initialize funding tracker.

        Args:
            accrual_interval_hours: How often to accrue funding (default: 1 hour)
            settlement_interval_hours: How often funding is settled (default: 8 hours)
        """
        self.accrual_interval_hours = accrual_interval_hours
        self.settlement_interval_hours = settlement_interval_hours

        # Per-symbol tracking
        self._accrued_funding: dict[str, Decimal] = {}
        self._last_accrual_ts: dict[str, datetime] = {}
        self._last_settlement_ts: dict[str, datetime] = {}

        # Historical record
        self._total_funding_paid: dict[str, Decimal] = {}  # Cumulative funding paid
        self._funding_events: list[FundingEvent] = []

    def accrue(
        self,
        symbol: str,
        position_size: Decimal,
        mark_price: Decimal,
        funding_rate_8h: Decimal,
        current_time: datetime,
    ) -> Decimal:
        """
        Accrue funding for the current period.

        Args:
            symbol: Trading pair (e.g., "BTC-PERP-USDC")
            position_size: Position size (positive for long, negative for short)
            mark_price: Current mark price
            funding_rate_8h: 8-hour funding rate (e.g., 0.0001 = 0.01%)
            current_time: Current timestamp

        Returns:
            Funding accrued this period (positive = paid, negative = received)

        Note:
            Funding rates are typically quoted as 8-hour rates. We pro-rate
            them to the accrual interval (e.g., 1 hour = 1/8 of the 8-hour rate).
        """
        # Initialize if first time seeing this symbol
        if symbol not in self._last_accrual_ts:
            self._last_accrual_ts[symbol] = current_time
            self._accrued_funding[symbol] = Decimal("0")
            self._total_funding_paid[symbol] = Decimal("0")
            return Decimal("0")

        # Check if enough time has passed for accrual
        time_since_last = current_time - self._last_accrual_ts[symbol]
        hours_elapsed = Decimal(str(time_since_last.total_seconds())) / Decimal("3600")

        if hours_elapsed < self.accrual_interval_hours:
            return Decimal("0")  # Not time to accrue yet

        # Calculate pro-rated funding for the interval
        # funding_rate_8h is the rate per 8 hours, so divide by 8 to get hourly
        hourly_rate = funding_rate_8h / Decimal("8")
        interval_rate = hourly_rate * Decimal(str(self.accrual_interval_hours))

        # Funding payment calculation
        # Long position (positive size): pay if rate is positive
        # Short position (negative size): receive if rate is positive
        funding_payment = position_size * mark_price * interval_rate

        # Update accrued funding
        self._accrued_funding[symbol] += funding_payment
        self._last_accrual_ts[symbol] = current_time

        return funding_payment

    def settle(
        self,
        symbol: str,
        current_time: datetime,
    ) -> Decimal:
        """
        Settle accrued funding and reset.

        This should be called at the funding settlement times (e.g., every 8 hours).

        Args:
            symbol: Trading pair
            current_time: Settlement timestamp

        Returns:
            Total funding settled (positive = paid, negative = received)
        """
        if symbol not in self._accrued_funding:
            return Decimal("0")

        # Check if settlement is due
        if symbol in self._last_settlement_ts:
            time_since_last = current_time - self._last_settlement_ts[symbol]
            hours_elapsed = time_since_last.total_seconds() / 3600
            if hours_elapsed < self.settlement_interval_hours:
                return Decimal("0")  # Not time to settle yet

        # Settle accrued funding
        settled = self._accrued_funding[symbol]
        self._accrued_funding[symbol] = Decimal("0")
        self._last_settlement_ts[symbol] = current_time

        # Record event
        self._total_funding_paid[symbol] += settled
        self._funding_events.append(
            FundingEvent(
                symbol=symbol,
                timestamp=current_time,
                amount=settled,
            )
        )

        return settled

    def get_accrued(self, symbol: str) -> Decimal:
        """Get current accrued funding for a symbol (not yet settled)."""
        return self._accrued_funding.get(symbol, Decimal("0"))

    def get_total_paid(self, symbol: str) -> Decimal:
        """Get total funding paid for a symbol (cumulative)."""
        return self._total_funding_paid.get(symbol, Decimal("0"))

    def get_total_funding_pnl(self) -> Decimal:
        """Get total funding PnL across all symbols."""
        return sum(self._total_funding_paid.values(), Decimal("0"))

    def get_funding_events(
        self,
        symbol: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list["FundingEvent"]:
        """
        Get historical funding events.

        Args:
            symbol: Filter by symbol (optional)
            start: Filter by start time (optional)
            end: Filter by end time (optional)

        Returns:
            List of funding events
        """
        events = self._funding_events

        if symbol:
            events = [e for e in events if e.symbol == symbol]

        if start:
            events = [e for e in events if e.timestamp >= start]

        if end:
            events = [e for e in events if e.timestamp < end]

        return events

    def should_settle(self, current_time: datetime, symbol: str) -> bool:
        """Check if funding should be settled for a symbol."""
        if symbol not in self._last_settlement_ts:
            return True  # First settlement

        time_since_last = current_time - self._last_settlement_ts[symbol]
        hours_elapsed = time_since_last.total_seconds() / 3600
        return hours_elapsed >= self.settlement_interval_hours


class FundingEvent:
    """Record of a funding settlement event."""

    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        amount: Decimal,
    ):
        self.symbol = symbol
        self.timestamp = timestamp
        self.amount = amount  # Positive = paid, negative = received

    def __repr__(self) -> str:
        direction = "paid" if self.amount > 0 else "received"
        return f"FundingEvent({self.symbol} {direction} {abs(self.amount)} at {self.timestamp})"
