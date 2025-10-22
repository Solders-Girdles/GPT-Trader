"""
Derivatives-enabled portfolio for backtesting.

Extends BacktestPortfolio with funding, margin, and liquidation tracking.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.strategies.decisions import Decision
from bot_v2.utilities.logging_patterns import get_logger

from .backtest_derivatives import FundingRateSimulator, LiquidationDetector, MarginTracker
from .backtest_portfolio import BacktestPortfolio
from .types_v2 import BacktestConfig, ExecutionResult

logger = get_logger(__name__, component="optimize")


class DerivativesBacktestPortfolio(BacktestPortfolio):
    """
    Portfolio simulator with derivatives features.

    Extends BacktestPortfolio with:
    - Funding rate simulation
    - Margin requirement tracking
    - Liquidation detection and enforcement
    """

    def __init__(
        self,
        *,
        initial_capital: Decimal,
        commission_rate: Decimal = Decimal("0.001"),
        slippage_rate: Decimal = Decimal("0.0005"),
        # Derivatives-specific parameters
        enable_funding: bool = True,
        funding_rate: Decimal | None = None,
        funding_interval_hours: int = 8,
        enable_margin_tracking: bool = True,
        initial_margin_rate: Decimal = Decimal("0.10"),  # 10% = 10x max
        maintenance_margin_rate: Decimal = Decimal("0.05"),  # 5% maintenance
        enable_margin_windows: bool = False,
        enable_liquidation: bool = True,
        liquidation_buffer_pct: Decimal = Decimal("0.15"),  # 15% warning buffer
        leverage: Decimal = Decimal("1"),  # Default leverage for sizing
    ):
        """
        Initialize derivatives portfolio.

        Args:
            initial_capital: Starting cash
            commission_rate: Commission rate
            slippage_rate: Slippage rate
            enable_funding: Enable funding rate simulation
            funding_rate: Fixed funding rate (None for dynamic)
            funding_interval_hours: Hours between funding payments
            enable_margin_tracking: Track margin requirements
            initial_margin_rate: Initial margin requirement rate
            maintenance_margin_rate: Maintenance margin rate
            enable_margin_windows: Enable time-based margin windows
            enable_liquidation: Enable liquidation detection
            liquidation_buffer_pct: Warning buffer distance
            leverage: Default leverage for position sizing
        """
        super().__init__(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )

        self.leverage = leverage

        # Derivatives components
        self.enable_funding = enable_funding
        self.enable_margin_tracking = enable_margin_tracking
        self.enable_liquidation = enable_liquidation

        if enable_funding:
            self.funding_simulator = FundingRateSimulator(
                funding_rate=funding_rate,
                funding_interval_hours=funding_interval_hours,
            )
            self.last_funding_time: datetime | None = None
        else:
            self.funding_simulator = None
            self.last_funding_time = None

        if enable_margin_tracking:
            self.margin_tracker = MarginTracker(
                initial_margin_rate=initial_margin_rate,
                maintenance_margin_rate=maintenance_margin_rate,
                enable_margin_windows=enable_margin_windows,
            )
        else:
            self.margin_tracker = None

        if enable_liquidation:
            self.liquidation_detector = LiquidationDetector(
                liquidation_buffer_pct=liquidation_buffer_pct,
                maintenance_margin_rate=maintenance_margin_rate,
            )
            self.liquidated_positions: list[dict[str, Any]] = []
        else:
            self.liquidation_detector = None
            self.liquidated_positions = []

    def process_decision(
        self,
        *,
        decision: Decision,
        symbol: str,
        current_price: Decimal,
        product: Product,
        timestamp: datetime,
    ) -> ExecutionResult:
        """
        Process decision with derivatives features.

        Overrides parent to add:
        1. Funding rate application before trade
        2. Liquidation checks before trade
        3. Margin requirement validation
        """
        # Apply funding if enabled
        if self.enable_funding and self.funding_simulator:
            self._maybe_apply_funding(timestamp=timestamp, current_prices={symbol: current_price})

        # Check for liquidations before processing new decision
        if self.enable_liquidation and self.liquidation_detector:
            self._check_and_apply_liquidations(timestamp=timestamp, current_prices={symbol: current_price})

        # Process decision normally
        execution = super().process_decision(
            decision=decision,
            symbol=symbol,
            current_price=current_price,
            product=product,
            timestamp=timestamp,
        )

        # Track margin state after execution
        if self.enable_margin_tracking and self.margin_tracker and execution.filled:
            self._record_margin_state(timestamp=timestamp, current_prices={symbol: current_price})

        return execution

    def _maybe_apply_funding(
        self,
        *,
        timestamp: datetime,
        current_prices: dict[str, Decimal],
        funding_rate_override: dict[str, Decimal] | None = None,
    ) -> None:
        """Apply funding if due."""
        if not self.funding_simulator:
            return

        if not self.positions:
            return

        # Check if funding is due
        if not self.funding_simulator.should_apply_funding(timestamp, self.last_funding_time):
            return

        # Apply funding
        funding_payment, events = self.funding_simulator.apply_funding(
            positions=self.positions,
            current_prices=current_prices,
            current_time=timestamp,
            funding_rate_override=funding_rate_override,
        )

        # Deduct from cash
        self.cash -= funding_payment

        # Update last funding time
        self.last_funding_time = timestamp

        logger.info(
            "Funding applied | timestamp=%s | payment=%s | positions=%d",
            timestamp,
            funding_payment,
            len(self.positions),
        )

    def _check_and_apply_liquidations(
        self, *, timestamp: datetime, current_prices: dict[str, Decimal]
    ) -> None:
        """Check all positions for liquidation and force-close if needed."""
        if not self.liquidation_detector:
            return

        symbols_to_liquidate = []

        for symbol, position in self.positions.items():
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            # Check liquidation
            should_liquidate, liq_price, warning = self.liquidation_detector.check_liquidation(
                symbol=symbol,
                position=position,
                current_price=current_price,
                current_time=timestamp,
                leverage=self.leverage,
            )

            if should_liquidate:
                symbols_to_liquidate.append((symbol, position, liq_price))

        # Force-close liquidated positions
        for symbol, position, liq_price in symbols_to_liquidate:
            self._force_liquidate_position(
                symbol=symbol, position=position, liq_price=liq_price, timestamp=timestamp
            )

    def _force_liquidate_position(
        self, *, symbol: str, position: dict[str, Any], liq_price: Decimal, timestamp: datetime
    ) -> None:
        """Force-close a liquidated position."""
        quantity = position["quantity"]
        side = position["side"]
        entry_price = position["entry"]

        # Liquidation occurs at liquidation price (not market)
        # Usually with penalty/slippage
        liquidation_penalty = Decimal("0.005")  # 0.5% liquidation fee

        if side == "long":
            proceeds = quantity * liq_price * (Decimal("1") - liquidation_penalty)
            self.cash += proceeds
            cost_basis = quantity * entry_price
            realized_pnl = proceeds - cost_basis
        else:  # short
            cost_to_cover = quantity * liq_price * (Decimal("1") + liquidation_penalty)
            self.cash -= cost_to_cover
            proceeds_at_entry = quantity * entry_price
            realized_pnl = proceeds_at_entry - cost_to_cover

        # Record liquidation
        self.liquidated_positions.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "liquidation_price": liq_price,
                "realized_pnl": realized_pnl,
            }
        )

        # Remove position
        del self.positions[symbol]
        self.trade_count += 1

        logger.warning(
            "LIQUIDATION EXECUTED | symbol=%s | side=%s | qty=%s | entry=%s | liq=%s | pnl=%s",
            symbol,
            side,
            quantity,
            entry_price,
            liq_price,
            realized_pnl,
        )

    def _record_margin_state(
        self, *, timestamp: datetime, current_prices: dict[str, Decimal]
    ) -> None:
        """Record current margin state."""
        if not self.margin_tracker:
            return

        # Calculate margin requirements
        margin_req = self.margin_tracker.calculate_margin_requirements(
            positions=self.positions,
            current_prices=current_prices,
            current_time=timestamp,
        )

        # Calculate utilization
        equity = self.get_equity(current_prices)
        utilization = self.margin_tracker.calculate_margin_utilization(
            margin_requirements=margin_req,
            equity=equity,
        )

        # Store snapshot
        snapshot = {**margin_req, **utilization}
        self.margin_tracker.margin_snapshots.append(snapshot)

    def get_derivatives_stats(self) -> dict[str, Any]:
        """Get derivatives-specific statistics."""
        stats = self.get_stats()

        # Add funding stats
        if self.funding_simulator:
            stats["total_funding_paid"] = self.funding_simulator.total_funding_paid
            stats["funding_payments_count"] = len(self.funding_simulator.funding_payments)

        # Add liquidation stats
        if self.liquidation_detector:
            stats["liquidation_count"] = len(self.liquidation_detector.liquidation_events)
            stats["liquidation_warnings"] = len(self.liquidation_detector.liquidation_warnings)
            stats["liquidated_positions"] = len(self.liquidated_positions)

        # Add margin stats
        if self.margin_tracker and self.margin_tracker.margin_snapshots:
            latest_margin = self.margin_tracker.margin_snapshots[-1]
            stats["current_leverage"] = float(latest_margin.get("leverage", 0))
            stats["margin_utilization"] = float(latest_margin.get("margin_utilization", 0))
            stats["free_margin"] = float(latest_margin.get("free_margin", 0))

        return stats

    def apply_funding_schedule(
        self, *, funding_schedule: dict[datetime, Decimal], timestamp: datetime, current_prices: dict[str, Decimal]
    ) -> None:
        """
        Apply funding from a predefined schedule (for stress testing).

        Args:
            funding_schedule: Dict mapping timestamp -> funding_rate
            timestamp: Current timestamp
            current_prices: Current prices
        """
        if not self.enable_funding or not self.funding_simulator:
            return

        # Check if this timestamp has a funding event
        if timestamp not in funding_schedule:
            return

        funding_rate = funding_schedule[timestamp]

        # Apply funding with override rate
        funding_override = {symbol: funding_rate for symbol in self.positions.keys()}

        self._maybe_apply_funding(
            timestamp=timestamp,
            current_prices=current_prices,
            funding_rate_override=funding_override,
        )


__all__ = ["DerivativesBacktestPortfolio"]
