"""
Derivatives-specific extensions for backtesting.

Adds funding rate simulation, margin tracking, and liquidation detection
to the production-parity backtesting framework.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="optimize")


class FundingRateSimulator:
    """
    Simulates funding rate payments for perpetual futures.

    Funding occurs at fixed intervals (typically every 8 hours at 00:00, 08:00, 16:00 UTC).
    """

    def __init__(
        self,
        *,
        funding_rate: Decimal | None = None,
        funding_interval_hours: int = 8,
    ):
        """
        Initialize funding simulator.

        Args:
            funding_rate: Fixed funding rate per interval (e.g., 0.0001 = 0.01%)
                         If None, will use dynamic rates from scenario
            funding_interval_hours: Hours between funding payments (default 8)
        """
        self.funding_rate = funding_rate or Decimal("0.0001")  # 0.01% default
        self.funding_interval_hours = funding_interval_hours
        self.funding_times = [0, 8, 16]  # UTC hours for funding
        self.total_funding_paid = Decimal("0")
        self.funding_payments: list[dict[str, Any]] = []

    def get_next_funding_time(self, current_time: datetime) -> datetime:
        """Get next funding time after current_time."""
        current_hour = current_time.hour

        # Find next funding hour
        next_hour = None
        for hour in self.funding_times:
            if hour > current_hour:
                next_hour = hour
                break

        if next_hour is None:
            # Next funding is tomorrow at first interval
            next_hour = self.funding_times[0]
            next_date = current_time.date() + timedelta(days=1)
        else:
            next_date = current_time.date()

        return datetime.combine(next_date, datetime.min.time()).replace(hour=next_hour)

    def should_apply_funding(
        self, current_time: datetime, last_funding_time: datetime | None
    ) -> bool:
        """Check if funding should be applied at current time."""
        if last_funding_time is None:
            # First funding: check if we've crossed a funding hour
            return current_time.hour in self.funding_times and current_time.minute == 0

        # Check if we've crossed a funding interval
        time_since_last = (current_time - last_funding_time).total_seconds() / 3600
        return time_since_last >= self.funding_interval_hours

    def calculate_funding_payment(
        self,
        *,
        position_size: Decimal,
        position_side: str,
        mark_price: Decimal,
        funding_rate: Decimal | None = None,
    ) -> Decimal:
        """
        Calculate funding payment for a position.

        Args:
            position_size: Position size (always positive)
            position_side: "long" or "short"
            mark_price: Current mark price
            funding_rate: Override funding rate (optional)

        Returns:
            Funding payment (positive = pay, negative = receive)
        """
        rate = funding_rate if funding_rate is not None else self.funding_rate

        notional = position_size * mark_price
        payment = notional * rate

        # Longs pay when rate is positive, shorts receive
        # Shorts pay when rate is negative, longs receive
        if position_side == "long":
            return payment
        else:  # short
            return -payment

    def apply_funding(
        self,
        *,
        positions: dict[str, dict[str, Any]],
        current_prices: dict[str, Decimal],
        current_time: datetime,
        funding_rate_override: dict[str, Decimal] | None = None,
    ) -> tuple[Decimal, list[dict[str, Any]]]:
        """
        Apply funding to all open positions.

        Args:
            positions: Current positions {symbol: position_state}
            current_prices: Current mark prices {symbol: price}
            current_time: Current timestamp
            funding_rate_override: Per-symbol funding rates (optional)

        Returns:
            (total_funding_paid, funding_events)
        """
        total_payment = Decimal("0")
        funding_events = []

        for symbol, position in positions.items():
            if symbol not in current_prices:
                logger.warning("Missing price for funding calculation | symbol=%s", symbol)
                continue

            position_size = position["quantity"]
            position_side = position["side"]
            mark_price = current_prices[symbol]

            # Get funding rate (use override if provided)
            rate = None
            if funding_rate_override and symbol in funding_rate_override:
                rate = funding_rate_override[symbol]

            payment = self.calculate_funding_payment(
                position_size=position_size,
                position_side=position_side,
                mark_price=mark_price,
                funding_rate=rate,
            )

            total_payment += payment

            funding_events.append(
                {
                    "timestamp": current_time,
                    "symbol": symbol,
                    "position_size": position_size,
                    "position_side": position_side,
                    "mark_price": mark_price,
                    "funding_rate": rate or self.funding_rate,
                    "payment": payment,
                }
            )

            logger.debug(
                "Funding applied | symbol=%s | side=%s | size=%s | rate=%s | payment=%s",
                symbol,
                position_side,
                position_size,
                rate or self.funding_rate,
                payment,
            )

        self.total_funding_paid += total_payment
        self.funding_payments.extend(funding_events)

        return total_payment, funding_events


class MarginTracker:
    """
    Tracks margin requirements and utilization for leveraged positions.

    Implements margin window policy similar to production.
    """

    def __init__(
        self,
        *,
        initial_margin_rate: Decimal = Decimal("0.10"),  # 10% = 10x leverage
        maintenance_margin_rate: Decimal = Decimal("0.05"),  # 5% maintenance
        enable_margin_windows: bool = False,
    ):
        """
        Initialize margin tracker.

        Args:
            initial_margin_rate: Initial margin requirement (1/max_leverage)
            maintenance_margin_rate: Maintenance margin requirement
            enable_margin_windows: Enable time-based margin window policy
        """
        self.initial_margin_rate = initial_margin_rate
        self.maintenance_margin_rate = maintenance_margin_rate
        self.enable_margin_windows = enable_margin_windows

        # Margin state history
        self.margin_snapshots: list[dict[str, Any]] = []

    def get_margin_rates(self, current_time: datetime) -> tuple[Decimal, Decimal]:
        """
        Get margin rates based on time window.

        Returns:
            (initial_margin_rate, maintenance_margin_rate)
        """
        if not self.enable_margin_windows:
            return self.initial_margin_rate, self.maintenance_margin_rate

        hour = current_time.hour

        # Margin window policy (UTC times)
        if 22 <= hour or hour < 6:
            # OVERNIGHT: 20% initial (5x max), 10% maintenance
            return Decimal("0.20"), Decimal("0.10")
        elif 14 <= hour < 16:
            # INTRADAY (high volatility): 15% initial (6.67x max), 7.5% maintenance
            return Decimal("0.15"), Decimal("0.075")
        else:
            # NORMAL: 10% initial (10x max), 5% maintenance
            return Decimal("0.10"), Decimal("0.05")

    def calculate_margin_requirements(
        self,
        *,
        positions: dict[str, dict[str, Any]],
        current_prices: dict[str, Decimal],
        current_time: datetime,
    ) -> dict[str, Any]:
        """
        Calculate margin requirements for current portfolio.

        Args:
            positions: Current positions
            current_prices: Current mark prices
            current_time: Current timestamp

        Returns:
            Dict with margin metrics
        """
        initial_rate, maintenance_rate = self.get_margin_rates(current_time)

        total_notional = Decimal("0")
        initial_margin_required = Decimal("0")
        maintenance_margin_required = Decimal("0")
        position_details = []

        for symbol, position in positions.items():
            if symbol not in current_prices:
                continue

            quantity = position["quantity"]
            mark_price = current_prices[symbol]
            notional = quantity * mark_price

            pos_initial = notional * initial_rate
            pos_maintenance = notional * maintenance_rate

            total_notional += notional
            initial_margin_required += pos_initial
            maintenance_margin_required += pos_maintenance

            position_details.append(
                {
                    "symbol": symbol,
                    "notional": notional,
                    "initial_margin": pos_initial,
                    "maintenance_margin": pos_maintenance,
                }
            )

        return {
            "timestamp": current_time,
            "total_notional": total_notional,
            "initial_margin_required": initial_margin_required,
            "maintenance_margin_required": maintenance_margin_required,
            "initial_margin_rate": initial_rate,
            "maintenance_margin_rate": maintenance_rate,
            "positions": position_details,
        }

    def calculate_margin_utilization(
        self,
        *,
        margin_requirements: dict[str, Any],
        equity: Decimal,
    ) -> dict[str, Any]:
        """
        Calculate margin utilization metrics.

        Args:
            margin_requirements: Output from calculate_margin_requirements
            equity: Current portfolio equity

        Returns:
            Dict with utilization metrics
        """
        initial_req = margin_requirements["initial_margin_required"]
        maintenance_req = margin_requirements["maintenance_margin_required"]
        notional = margin_requirements["total_notional"]

        # Calculate metrics
        if equity > Decimal("0"):
            margin_utilization = initial_req / equity
            maintenance_utilization = maintenance_req / equity
            leverage = notional / equity
        else:
            margin_utilization = Decimal("0")
            maintenance_utilization = Decimal("0")
            leverage = Decimal("0")

        # Calculate free margin
        free_margin = equity - maintenance_req

        return {
            "equity": equity,
            "margin_utilization": margin_utilization,
            "maintenance_utilization": maintenance_utilization,
            "leverage": leverage,
            "free_margin": free_margin,
            "at_risk": free_margin < Decimal("0"),
        }


class LiquidationDetector:
    """
    Detects liquidation events during backtesting.

    Monitors position liquidation prices and triggers liquidations when
    mark price crosses the threshold.
    """

    def __init__(
        self,
        *,
        liquidation_buffer_pct: Decimal = Decimal("0.15"),  # 15% buffer
        maintenance_margin_rate: Decimal = Decimal("0.05"),  # 5% maintenance
    ):
        """
        Initialize liquidation detector.

        Args:
            liquidation_buffer_pct: Buffer distance to trigger warnings
            maintenance_margin_rate: Maintenance margin rate for calculations
        """
        self.liquidation_buffer_pct = liquidation_buffer_pct
        self.maintenance_margin_rate = maintenance_margin_rate
        self.liquidation_events: list[dict[str, Any]] = []
        self.liquidation_warnings: list[dict[str, Any]] = []

    def calculate_liquidation_price(
        self,
        *,
        entry_price: Decimal,
        position_side: str,
        leverage: Decimal,
        maintenance_margin_rate: Decimal | None = None,
    ) -> Decimal:
        """
        Calculate liquidation price for a position.

        Args:
            entry_price: Position entry price
            position_side: "long" or "short"
            leverage: Position leverage
            maintenance_margin_rate: Override maintenance margin rate

        Returns:
            Liquidation price
        """
        mmr = maintenance_margin_rate or self.maintenance_margin_rate

        if position_side == "long":
            # Long liquidation: price falls below entry - (entry/leverage) + (entry * mmr)
            liq_price = entry_price * (Decimal("1") - Decimal("1") / leverage + mmr)
        else:  # short
            # Short liquidation: price rises above entry + (entry/leverage) - (entry * mmr)
            liq_price = entry_price * (Decimal("1") + Decimal("1") / leverage - mmr)

        return liq_price

    def calculate_liquidation_distance(
        self, *, current_price: Decimal, liquidation_price: Decimal, position_side: str
    ) -> Decimal:
        """
        Calculate distance to liquidation as percentage.

        Args:
            current_price: Current mark price
            liquidation_price: Liquidation price
            position_side: "long" or "short"

        Returns:
            Distance as decimal (0.15 = 15% away from liquidation)
        """
        if position_side == "long":
            # For longs, liquidation is below current price
            distance = (current_price - liquidation_price) / current_price
        else:  # short
            # For shorts, liquidation is above current price
            distance = (liquidation_price - current_price) / current_price

        return distance

    def check_liquidation(
        self,
        *,
        symbol: str,
        position: dict[str, Any],
        current_price: Decimal,
        current_time: datetime,
        leverage: Decimal,
    ) -> tuple[bool, Decimal | None, str | None]:
        """
        Check if position should be liquidated.

        Args:
            symbol: Trading symbol
            position: Position state
            current_price: Current mark price
            current_time: Current timestamp
            leverage: Position leverage

        Returns:
            (should_liquidate, liquidation_price, warning_level)
        """
        entry_price = position["entry"]
        position_side = position["side"]

        # Calculate liquidation price
        liq_price = self.calculate_liquidation_price(
            entry_price=entry_price, position_side=position_side, leverage=leverage
        )

        # Calculate distance
        distance = self.calculate_liquidation_distance(
            current_price=current_price, liquidation_price=liq_price, position_side=position_side
        )

        # Determine status
        should_liquidate = distance <= Decimal("0")
        warning_level = None

        if should_liquidate:
            warning_level = "LIQUIDATED"
            self.liquidation_events.append(
                {
                    "timestamp": current_time,
                    "symbol": symbol,
                    "position_side": position_side,
                    "entry_price": entry_price,
                    "liquidation_price": liq_price,
                    "trigger_price": current_price,
                    "quantity": position["quantity"],
                }
            )
            logger.warning(
                "LIQUIDATION TRIGGERED | symbol=%s | side=%s | entry=%s | liq=%s | current=%s",
                symbol,
                position_side,
                entry_price,
                liq_price,
                current_price,
            )
        elif distance <= self.liquidation_buffer_pct:
            warning_level = "CRITICAL" if distance <= Decimal("0.05") else "WARNING"
            self.liquidation_warnings.append(
                {
                    "timestamp": current_time,
                    "symbol": symbol,
                    "position_side": position_side,
                    "liquidation_price": liq_price,
                    "current_price": current_price,
                    "distance_pct": float(distance),
                    "warning_level": warning_level,
                }
            )

        return should_liquidate, liq_price, warning_level


__all__ = ["FundingRateSimulator", "MarginTracker", "LiquidationDetector"]
