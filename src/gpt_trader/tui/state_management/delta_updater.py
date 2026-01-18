"""
State Delta Updater.

Provides efficient delta-based state updates to minimize UI flicker
and unnecessary re-renders. Instead of replacing entire state objects,
this module compares old and new values and only updates what changed.

This is particularly important during mode switches and reconnections
where full state replacement can cause visible flicker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.types import (
        AccountSummary,
        ActiveOrders,
        MarketState,
        PortfolioSummary,
        RiskState,
        StrategyState,
        SystemStatus,
        TradeHistory,
    )

logger = get_logger(__name__, component="tui")


@dataclass
class DeltaResult:
    """Result of a delta comparison.

    Attributes:
        has_changes: True if any changes were detected
        changed_fields: List of field names that changed
        details: Dict of field -> (old_value, new_value) for changed fields
    """

    has_changes: bool = False
    changed_fields: list[str] = field(default_factory=list)
    details: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def add_change(self, field_name: str, old_value: Any, new_value: Any) -> None:
        """Record a changed field."""
        self.has_changes = True
        self.changed_fields.append(field_name)
        self.details[field_name] = (old_value, new_value)


class StateDeltaUpdater:
    """Computes and applies delta updates to TUI state.

    Instead of replacing entire state objects (which triggers full re-renders),
    this class compares old and new values and only updates what changed.

    This reduces UI flicker and improves perceived responsiveness,
    especially during mode switches or data reconnections.

    Usage:
        delta_updater = StateDeltaUpdater()

        # Check if update is needed
        result = delta_updater.compare_market(old_market, new_market)
        if result.has_changes:
            # Apply only changed values
            delta_updater.apply_market_delta(state, new_market, result)
    """

    # Threshold for floating point comparison
    FLOAT_EPSILON = 1e-9

    # Threshold for Decimal comparison (8 decimal places)
    DECIMAL_EPSILON = Decimal("0.00000001")

    def compare_market(self, old: MarketState, new: MarketState) -> DeltaResult:
        """Compare market state for changes.

        Args:
            old: Current market state
            new: New market state

        Returns:
            DeltaResult indicating what changed
        """
        result = DeltaResult()

        # Compare prices
        old_prices = old.prices if old else {}
        new_prices = new.prices if new else {}

        # Check for new or changed prices
        for symbol, new_price in new_prices.items():
            old_price = old_prices.get(symbol)
            if old_price is None or not self._decimal_equal(old_price, new_price):
                result.add_change(f"prices.{symbol}", old_price, new_price)

        # Check for removed prices
        for symbol in old_prices:
            if symbol not in new_prices:
                result.add_change(f"prices.{symbol}", old_prices[symbol], None)

        # Compare last_update timestamp
        old_update = old.last_update if old else 0.0
        new_update = new.last_update if new else 0.0
        if not self._float_equal(old_update, new_update):
            result.add_change("last_update", old_update, new_update)

        # Compare price history (just check if changed, don't detail)
        old_history = old.price_history if old else {}
        new_history = new.price_history if new else {}
        if old_history != new_history:
            result.add_change("price_history", "[old]", "[new]")

        return result

    def compare_positions(self, old: PortfolioSummary, new: PortfolioSummary) -> DeltaResult:
        """Compare position state for changes."""
        result = DeltaResult()

        old_positions = old.positions if old else {}
        new_positions = new.positions if new else {}

        # Check for new or changed positions
        for symbol, new_pos in new_positions.items():
            old_pos = old_positions.get(symbol)
            if old_pos is None:
                result.add_change(f"positions.{symbol}", None, new_pos)
            else:
                # Compare position fields
                if not self._decimal_equal(old_pos.quantity, new_pos.quantity):
                    result.add_change(
                        f"positions.{symbol}.quantity",
                        old_pos.quantity,
                        new_pos.quantity,
                    )
                if not self._decimal_equal(old_pos.unrealized_pnl, new_pos.unrealized_pnl):
                    result.add_change(
                        f"positions.{symbol}.unrealized_pnl",
                        old_pos.unrealized_pnl,
                        new_pos.unrealized_pnl,
                    )
                if not self._decimal_equal(old_pos.mark_price, new_pos.mark_price):
                    result.add_change(
                        f"positions.{symbol}.mark_price",
                        old_pos.mark_price,
                        new_pos.mark_price,
                    )

        # Check for closed positions
        for symbol in old_positions:
            if symbol not in new_positions:
                result.add_change(f"positions.{symbol}", old_positions[symbol], None)

        # Compare totals
        if not self._decimal_equal(old.total_unrealized_pnl, new.total_unrealized_pnl):
            result.add_change(
                "total_unrealized_pnl",
                old.total_unrealized_pnl,
                new.total_unrealized_pnl,
            )

        if not self._decimal_equal(old.equity, new.equity):
            result.add_change("equity", old.equity, new.equity)

        return result

    def compare_orders(self, old: ActiveOrders, new: ActiveOrders) -> DeltaResult:
        """Compare order state for changes."""
        result = DeltaResult()

        old_orders = {o.order_id: o for o in (old.orders if old else [])}
        new_orders = {o.order_id: o for o in (new.orders if new else [])}

        # Check for new or changed orders
        for order_id, new_order in new_orders.items():
            old_order = old_orders.get(order_id)
            if old_order is None:
                result.add_change(f"orders.{order_id}", None, new_order)
            elif old_order.status != new_order.status:
                result.add_change(
                    f"orders.{order_id}.status",
                    old_order.status,
                    new_order.status,
                )

        # Check for removed orders
        for order_id in old_orders:
            if order_id not in new_orders:
                result.add_change(f"orders.{order_id}", old_orders[order_id], None)

        return result

    def compare_trades(self, old: TradeHistory, new: TradeHistory) -> DeltaResult:
        """Compare trade history for changes."""
        result = DeltaResult()

        old_trades = old.trades if old else []
        new_trades = new.trades if new else []

        # Compare by count first (fast path)
        if len(old_trades) != len(new_trades):
            result.add_change("trades.count", len(old_trades), len(new_trades))

        # Check for new trades (trades are append-only)
        old_trade_ids = {t.trade_id for t in old_trades}
        for trade in new_trades:
            if trade.trade_id not in old_trade_ids:
                result.add_change(f"trades.{trade.trade_id}", None, trade)

        return result

    def compare_account(self, old: AccountSummary, new: AccountSummary) -> DeltaResult:
        """Compare account state for changes."""
        result = DeltaResult()

        # Compare volume and fees
        if not self._decimal_equal(old.volume_30d, new.volume_30d):
            result.add_change("volume_30d", old.volume_30d, new.volume_30d)

        if not self._decimal_equal(old.fees_30d, new.fees_30d):
            result.add_change("fees_30d", old.fees_30d, new.fees_30d)

        if old.fee_tier != new.fee_tier:
            result.add_change("fee_tier", old.fee_tier, new.fee_tier)

        # Compare balances
        old_balances = {b.asset: b for b in old.balances}
        new_balances = {b.asset: b for b in new.balances}

        for asset, new_bal in new_balances.items():
            old_bal = old_balances.get(asset)
            if old_bal is None:
                result.add_change(f"balances.{asset}", None, new_bal)
            else:
                if not self._decimal_equal(old_bal.total, new_bal.total):
                    result.add_change(
                        f"balances.{asset}.total",
                        old_bal.total,
                        new_bal.total,
                    )
                if not self._decimal_equal(old_bal.available, new_bal.available):
                    result.add_change(
                        f"balances.{asset}.available",
                        old_bal.available,
                        new_bal.available,
                    )

        return result

    def compare_risk(self, old: RiskState, new: RiskState) -> DeltaResult:
        """Compare risk state for changes."""
        result = DeltaResult()

        if not self._float_equal(old.max_leverage, new.max_leverage):
            result.add_change("max_leverage", old.max_leverage, new.max_leverage)

        if not self._float_equal(old.daily_loss_limit_pct, new.daily_loss_limit_pct):
            result.add_change(
                "daily_loss_limit_pct",
                old.daily_loss_limit_pct,
                new.daily_loss_limit_pct,
            )

        if not self._float_equal(old.current_daily_loss_pct, new.current_daily_loss_pct):
            result.add_change(
                "current_daily_loss_pct",
                old.current_daily_loss_pct,
                new.current_daily_loss_pct,
            )

        if old.reduce_only_mode != new.reduce_only_mode:
            result.add_change(
                "reduce_only_mode",
                old.reduce_only_mode,
                new.reduce_only_mode,
            )

        if old.reduce_only_reason != new.reduce_only_reason:
            result.add_change(
                "reduce_only_reason",
                old.reduce_only_reason,
                new.reduce_only_reason,
            )

        if old.guards != new.guards:
            result.add_change("guards", old.guards, new.guards)

        return result

    def compare_system(self, old: SystemStatus, new: SystemStatus) -> DeltaResult:
        """Compare system status for changes."""
        result = DeltaResult()

        if not self._float_equal(old.api_latency, new.api_latency):
            result.add_change("api_latency", old.api_latency, new.api_latency)

        if old.connection_status != new.connection_status:
            result.add_change(
                "connection_status",
                old.connection_status,
                new.connection_status,
            )

        if old.rate_limit_usage != new.rate_limit_usage:
            result.add_change(
                "rate_limit_usage",
                old.rate_limit_usage,
                new.rate_limit_usage,
            )

        if old.memory_usage != new.memory_usage:
            result.add_change("memory_usage", old.memory_usage, new.memory_usage)

        if old.cpu_usage != new.cpu_usage:
            result.add_change("cpu_usage", old.cpu_usage, new.cpu_usage)

        return result

    def compare_strategy(self, old: StrategyState, new: StrategyState) -> DeltaResult:
        """Compare strategy state for changes."""
        result = DeltaResult()

        if old.active_strategies != new.active_strategies:
            result.add_change(
                "active_strategies",
                old.active_strategies,
                new.active_strategies,
            )

        # Compare decisions
        old_decisions = old.last_decisions if old else {}
        new_decisions = new.last_decisions if new else {}

        for symbol, new_dec in new_decisions.items():
            old_dec = old_decisions.get(symbol)
            if old_dec is None:
                result.add_change(f"decisions.{symbol}", None, new_dec)
            else:
                if old_dec.action != new_dec.action:
                    result.add_change(
                        f"decisions.{symbol}.action",
                        old_dec.action,
                        new_dec.action,
                    )
                if not self._float_equal(old_dec.confidence, new_dec.confidence):
                    result.add_change(
                        f"decisions.{symbol}.confidence",
                        old_dec.confidence,
                        new_dec.confidence,
                    )

        return result

    def should_update_component(self, component: str, delta: DeltaResult) -> bool:
        """Determine if a component needs updating based on delta.

        Args:
            component: Component name ("market", "positions", etc.)
            delta: The delta result from comparison

        Returns:
            True if the component should be updated
        """
        if not delta.has_changes:
            return False

        # For some components, minor changes don't warrant update
        # This can be extended for fine-grained control
        if component == "market":
            # Only update if prices or significant data changed
            significant_fields = ["prices", "last_update"]
            return any(
                field.startswith(tuple(significant_fields)) for field in delta.changed_fields
            )

        # Default: update if any changes
        return True

    def _float_equal(self, a: float, b: float) -> bool:
        """Compare floats with epsilon tolerance."""
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs(a - b) < self.FLOAT_EPSILON

    def _decimal_equal(self, a: Decimal | None, b: Decimal | None) -> bool:
        """Compare Decimals with epsilon tolerance."""
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        try:
            return abs(a - b) < self.DECIMAL_EPSILON
        except (TypeError, ValueError):
            return False
