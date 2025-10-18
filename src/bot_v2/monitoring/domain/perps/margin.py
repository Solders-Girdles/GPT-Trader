"""
Margin State Monitor for Production Trading.

Monitors margin requirements, windows, and usage for Coinbase derivatives
with support for day/overnight/intraday margin transitions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="margin_monitor")


class MarginWindow(Enum):
    """Coinbase margin window states."""

    NORMAL = "normal"  # Regular trading hours
    INTRADAY = "intraday"  # Tighter requirements during volatile periods
    OVERNIGHT = "overnight"  # Higher requirements outside market hours
    PRE_FUNDING = "pre_funding"  # Special window before funding


@dataclass
class MarginRequirement:
    """Margin requirement for a specific window."""

    initial_rate: Decimal  # Initial margin rate (e.g., 0.10 = 10%)
    maintenance_rate: Decimal  # Maintenance margin rate (e.g., 0.05 = 5%)
    max_leverage: Decimal  # Maximum allowed leverage
    window: MarginWindow


@dataclass
class MarginSnapshot:
    """Complete margin state snapshot."""

    timestamp: datetime
    window: MarginWindow

    # Balances
    total_equity: Decimal
    cash_balance: Decimal
    positions_notional: Decimal

    # Margin calculations
    initial_margin_req: Decimal
    maintenance_margin_req: Decimal
    margin_used: Decimal
    margin_available: Decimal

    # Utilization
    margin_utilization: Decimal  # margin_used / total_equity
    leverage: Decimal  # positions_notional / total_equity

    # Status
    is_margin_call: bool = False
    is_liquidation_risk: bool = False

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "window": self.window.value,
            "total_equity": float(self.total_equity),
            "cash_balance": float(self.cash_balance),
            "positions_notional": float(self.positions_notional),
            "initial_margin_req": float(self.initial_margin_req),
            "maintenance_margin_req": float(self.maintenance_margin_req),
            "margin_used": float(self.margin_used),
            "margin_available": float(self.margin_available),
            "margin_utilization": float(self.margin_utilization),
            "leverage": float(self.leverage),
            "is_margin_call": self.is_margin_call,
            "is_liquidation_risk": self.is_liquidation_risk,
        }


class MarginWindowPolicy:
    """
    Manages margin window transitions and policy adjustments.

    Determines current margin window based on time, market conditions,
    and funding schedules. Adjusts trading policies accordingly.
    """

    # Standard margin requirements by window
    MARGIN_REQUIREMENTS = {
        MarginWindow.NORMAL: MarginRequirement(
            initial_rate=Decimal("0.10"),  # 10% initial = 10x leverage
            maintenance_rate=Decimal("0.05"),  # 5% maintenance
            max_leverage=Decimal("10"),
            window=MarginWindow.NORMAL,
        ),
        MarginWindow.INTRADAY: MarginRequirement(
            initial_rate=Decimal("0.15"),  # 15% initial = 6.67x leverage
            maintenance_rate=Decimal("0.075"),  # 7.5% maintenance
            max_leverage=Decimal("6.67"),
            window=MarginWindow.INTRADAY,
        ),
        MarginWindow.OVERNIGHT: MarginRequirement(
            initial_rate=Decimal("0.20"),  # 20% initial = 5x leverage
            maintenance_rate=Decimal("0.10"),  # 10% maintenance
            max_leverage=Decimal("5"),
            window=MarginWindow.OVERNIGHT,
        ),
        MarginWindow.PRE_FUNDING: MarginRequirement(
            initial_rate=Decimal("0.25"),  # 25% initial = 4x leverage
            maintenance_rate=Decimal("0.125"),  # 12.5% maintenance
            max_leverage=Decimal("4"),
            window=MarginWindow.PRE_FUNDING,
        ),
    }

    def __init__(self) -> None:
        self._funding_times = [time(0, 0), time(8, 0), time(16, 0)]  # UTC funding times
        self._current_window = MarginWindow.NORMAL
        self._next_window_change: datetime | None = None

    def determine_current_window(self, current_time: datetime | None = None) -> MarginWindow:
        """
        Determine current margin window based on time and conditions.
        """
        if current_time is None:
            current_time = datetime.utcnow()

        current_hour = current_time.hour
        current_minute = current_time.minute

        # Check if within pre-funding window (30 minutes before funding)
        for funding_time in self._funding_times:
            funding_hour, funding_minute = funding_time.hour, funding_time.minute

            # Calculate minutes until funding
            if funding_hour == current_hour:
                minutes_until = funding_minute - current_minute
                if 0 <= minutes_until <= 30:
                    return MarginWindow.PRE_FUNDING

            # Handle wrap-around (e.g., 23:45 to 00:00)
            elif funding_hour == 0 and current_hour == 23 and current_minute >= 30:
                return MarginWindow.PRE_FUNDING

        # Check for overnight window (outside typical trading hours)
        # Assume overnight is 22:00 UTC to 06:00 UTC
        if current_hour >= 22 or current_hour < 6:
            return MarginWindow.OVERNIGHT

        # Check for intraday tightening (high volatility periods)
        # This would integrate with volatility monitoring
        # For now, assume 14:00-16:00 UTC (US market close volatility)
        if 14 <= current_hour < 16:
            return MarginWindow.INTRADAY

        return MarginWindow.NORMAL

    def get_requirements(self, window: MarginWindow) -> MarginRequirement:
        """Get margin requirements for specific window."""
        return self.MARGIN_REQUIREMENTS[window]

    def calculate_next_window_change(self, current_time: datetime | None = None) -> datetime:
        """Calculate when the next margin window change will occur."""
        if current_time is None:
            current_time = datetime.utcnow()

        # Find next significant time boundary
        next_changes = []

        # Next funding time
        for funding_time in self._funding_times:
            next_funding = current_time.replace(
                hour=funding_time.hour, minute=funding_time.minute, second=0, microsecond=0
            )
            if next_funding <= current_time:
                next_funding += timedelta(days=1)
            next_changes.append(next_funding)

        # Overnight window changes (22:00 and 06:00)
        for hour in [6, 22]:
            next_change = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            if next_change <= current_time:
                next_change += timedelta(days=1)
            next_changes.append(next_change)

        # Intraday window changes (14:00 and 16:00)
        for hour in [14, 16]:
            next_change = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            if next_change <= current_time:
                next_change += timedelta(days=1)
            next_changes.append(next_change)

        return min(next_changes)

    def should_reduce_risk(self, current_window: MarginWindow, next_window: MarginWindow) -> bool:
        """Check if risk should be reduced for upcoming window change."""
        current_req = self.get_requirements(current_window)
        next_req = self.get_requirements(next_window)

        # Reduce risk if next window has tighter requirements
        return next_req.initial_rate > current_req.initial_rate


class MarginStateMonitor:
    """
    Production margin state monitor.

    Tracks margin usage, requirements, and window transitions.
    Provides alerts and risk reduction recommendations.
    """

    def __init__(
        self,
        client: Any | None = None,
        alert_threshold: Decimal = Decimal("0.8"),  # 80% utilization alert
        liquidation_buffer: Decimal = Decimal("0.1"),  # 10% buffer before liquidation
    ) -> None:
        self.client = client
        self.policy = MarginWindowPolicy()
        self.alert_threshold = alert_threshold
        self.liquidation_buffer = liquidation_buffer

        # State tracking
        self._current_snapshot: MarginSnapshot | None = None
        self._snapshot_history: list[MarginSnapshot] = []
        self._alert_callbacks: list[Callable[[MarginSnapshot, str], Awaitable[None]]] = []

        logger.info(
            "MarginStateMonitor initialized",
            operation="margin_monitor",
            stage="init",
            alert_threshold=float(alert_threshold),
            liquidation_buffer=float(liquidation_buffer),
            client_provided=bool(client),
        )

    def add_alert_callback(
        self, callback: Callable[[MarginSnapshot, str], Awaitable[None]]
    ) -> None:
        """Add callback for margin alerts."""
        self._alert_callbacks.append(callback)

    async def compute_margin_state(
        self,
        total_equity: Decimal,
        cash_balance: Decimal,
        positions: Mapping[str, Mapping[str, Any]],
    ) -> MarginSnapshot:
        """
        Compute current margin state.

        Args:
            total_equity: Total account equity
            cash_balance: Available cash
            positions: Position data with mark prices

        Returns:
            Current margin snapshot
        """
        now = datetime.utcnow()
        current_window = self.policy.determine_current_window(now)
        requirements = self.policy.get_requirements(current_window)

        # Calculate total positions notional
        positions_notional = Decimal("0")
        for symbol, pos_data in positions.items():
            if pos_data.get("quantity", 0) != 0:
                quantity = abs(Decimal(str(pos_data["quantity"])))
                mark_price = Decimal(str(pos_data.get("mark_price", 0)))
                positions_notional += quantity * mark_price

        # Calculate margin requirements
        initial_margin_req = positions_notional * requirements.initial_rate
        maintenance_margin_req = positions_notional * requirements.maintenance_rate

        # Current margin usage (initial margin required)
        margin_used = initial_margin_req
        margin_available = max(Decimal("0"), total_equity - margin_used)

        # Utilization metrics
        margin_utilization = margin_used / total_equity if total_equity > 0 else Decimal("0")
        leverage = positions_notional / total_equity if total_equity > 0 else Decimal("0")

        # Risk assessment
        is_margin_call = total_equity <= maintenance_margin_req
        is_liquidation_risk = (total_equity - maintenance_margin_req) <= (
            maintenance_margin_req * self.liquidation_buffer
        )

        snapshot = MarginSnapshot(
            timestamp=now,
            window=current_window,
            total_equity=total_equity,
            cash_balance=cash_balance,
            positions_notional=positions_notional,
            initial_margin_req=initial_margin_req,
            maintenance_margin_req=maintenance_margin_req,
            margin_used=margin_used,
            margin_available=margin_available,
            margin_utilization=margin_utilization,
            leverage=leverage,
            is_margin_call=is_margin_call,
            is_liquidation_risk=is_liquidation_risk,
        )

        # Check for alerts
        await self._check_alerts(snapshot)

        # Store snapshot
        self._current_snapshot = snapshot
        self._store_snapshot(snapshot)

        return snapshot

    async def check_window_transition(self) -> dict[str, Any] | None:
        """
        Check for upcoming margin window transitions.

        Returns:
            Dict with transition info if change is imminent, None otherwise
        """
        current_time = datetime.utcnow()
        current_window = self.policy.determine_current_window(current_time)

        # Check 30 minutes ahead
        future_time = current_time + timedelta(minutes=30)
        future_window = self.policy.determine_current_window(future_time)

        if current_window != future_window:
            current_req = self.policy.get_requirements(current_window)
            future_req = self.policy.get_requirements(future_window)

            return {
                "current_window": current_window.value,
                "next_window": future_window.value,
                "change_time": future_time.isoformat(),
                "minutes_until": 30,
                "current_max_leverage": float(current_req.max_leverage),
                "next_max_leverage": float(future_req.max_leverage),
                "should_reduce_risk": self.policy.should_reduce_risk(current_window, future_window),
                "leverage_reduction_factor": (
                    float(future_req.max_leverage / current_req.max_leverage)
                    if current_req.max_leverage > 0
                    else 1.0
                ),
            }

        return None

    async def get_max_position_size(
        self, symbol: str, price: Decimal, available_equity: Decimal | None = None
    ) -> Decimal:
        """
        Calculate maximum position size given current margin requirements.

        Args:
            symbol: Trading symbol
            price: Entry price
            available_equity: Available equity (uses current if None)

        Returns:
            Maximum position size in base units
        """
        if available_equity is None:
            if not self._current_snapshot:
                return Decimal("0")
            available_equity = self._current_snapshot.margin_available

        current_window = self.policy.determine_current_window()
        requirements = self.policy.get_requirements(current_window)

        # Calculate max notional based on initial margin requirement
        max_notional = available_equity / requirements.initial_rate
        max_quantity = max_notional / price

        return max_quantity

    def get_current_state(self) -> MarginSnapshot | None:
        """Get current margin snapshot."""
        return self._current_snapshot

    def get_margin_history(self, hours_back: int = 24) -> list[dict[str, Any]]:
        """Get margin state history."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

        return [
            snapshot.to_dict()
            for snapshot in self._snapshot_history
            if snapshot.timestamp >= cutoff_time
        ]

    def get_current_window(self) -> MarginWindow:
        """Get the current margin window."""
        return self.policy.determine_current_window()

    async def _check_alerts(self, snapshot: MarginSnapshot) -> None:
        """Check for margin alerts and trigger callbacks."""
        alerts = []

        if snapshot.is_liquidation_risk:
            alerts.append("LIQUIDATION_RISK")
        elif snapshot.is_margin_call:
            alerts.append("MARGIN_CALL")
        elif snapshot.margin_utilization >= self.alert_threshold:
            alerts.append("HIGH_UTILIZATION")

        # Window change alerts
        transition = await self.check_window_transition()
        if transition and transition["should_reduce_risk"]:
            alerts.append("WINDOW_TRANSITION")

        # Trigger callbacks
        for alert_type in alerts:
            for callback in self._alert_callbacks:
                try:
                    await callback(snapshot, alert_type)
                except Exception as e:
                    logger.error(
                        "Margin alert callback failed",
                        operation="margin_monitor",
                        stage="alert_callback",
                        alert_type=alert_type,
                        error=str(e),
                        exc_info=True,
                    )

    def _store_snapshot(self, snapshot: MarginSnapshot) -> None:
        """Store snapshot and manage retention."""
        self._snapshot_history.append(snapshot)

        # Keep last 24 hours (assuming 5-minute snapshots = 288 snapshots)
        max_snapshots = 288
        if len(self._snapshot_history) > max_snapshots:
            self._snapshot_history = self._snapshot_history[-max_snapshots:]


async def create_margin_monitor(client: Any | None = None, **kwargs: Any) -> MarginStateMonitor:
    """Create and initialize margin state monitor."""
    monitor = MarginStateMonitor(client=client, **kwargs)
    logger.info(
        "MarginStateMonitor ready",
        operation="margin_monitor",
        stage="factory",
        client_provided=bool(client),
    )
    return monitor
