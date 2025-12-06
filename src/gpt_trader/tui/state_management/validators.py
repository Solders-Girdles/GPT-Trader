"""
State Validation Layer.

Provides validation for incoming BotStatus updates before they are applied
to the TUI state. Catches malformed data, out-of-range values, and
inconsistencies early to prevent UI corruption.

Validation errors are collected and reported via events, allowing the UI
to continue operating with partial data while alerting the user to issues.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.monitoring.status_reporter import BotStatus

logger = get_logger(__name__, component="tui")


@dataclass
class ValidationError:
    """Details of a single validation error.

    Attributes:
        field: Field path that failed validation (e.g., "market.prices.BTC-USD")
        message: Human-readable error message
        severity: "warning" for non-fatal, "error" for critical
        value: The invalid value (for debugging)
    """

    field: str
    message: str
    severity: str = "error"
    value: Any = None


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        valid: True if no errors found
        errors: List of validation errors
        warnings: List of validation warnings (non-blocking)
    """

    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def add_error(self, field_name: str, message: str, value: Any = None) -> None:
        """Add a validation error."""
        self.valid = False
        self.errors.append(
            ValidationError(
                field=field_name,
                message=message,
                severity="error",
                value=value,
            )
        )

    def add_warning(self, field_name: str, message: str, value: Any = None) -> None:
        """Add a validation warning (does not fail validation)."""
        self.warnings.append(
            ValidationError(
                field=field_name,
                message=message,
                severity="warning",
                value=value,
            )
        )

    def merge(self, other: ValidationResult) -> None:
        """Merge another validation result into this one."""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class StateValidator:
    """Validates incoming BotStatus data before state updates.

    Performs comprehensive validation on all state components:
    - Type checking for required fields
    - Range validation for numeric values
    - Consistency checks between related fields
    - Decimal parsing validation

    Usage:
        validator = StateValidator()
        result = validator.validate_full_state(bot_status)
        if not result.valid:
            # Handle errors
            for error in result.errors:
                logger.error(f"{error.field}: {error.message}")
    """

    # Validation thresholds
    MAX_PRICE = Decimal("1000000000")  # $1B max price
    MIN_PRICE = Decimal("0")
    MAX_QUANTITY = Decimal("1000000000")  # 1B max quantity
    MAX_LEVERAGE = 100.0  # Max supported leverage
    MAX_PNL_PERCENTAGE = 1000.0  # 1000% max daily P&L

    def validate_full_state(self, status: BotStatus) -> ValidationResult:
        """Validate all components of a BotStatus update.

        Args:
            status: The BotStatus to validate

        Returns:
            ValidationResult with any errors or warnings found
        """
        result = ValidationResult()

        # Validate each component independently
        try:
            result.merge(self._validate_market(status.market))
        except Exception as e:
            result.add_error("market", f"Market validation failed: {e}")

        try:
            result.merge(self._validate_positions(status.positions))
        except Exception as e:
            result.add_error("positions", f"Position validation failed: {e}")

        try:
            result.merge(self._validate_orders(status.orders))
        except Exception as e:
            result.add_error("orders", f"Order validation failed: {e}")

        try:
            result.merge(self._validate_trades(status.trades))
        except Exception as e:
            result.add_error("trades", f"Trade validation failed: {e}")

        try:
            result.merge(self._validate_account(status.account))
        except Exception as e:
            result.add_error("account", f"Account validation failed: {e}")

        try:
            result.merge(self._validate_risk(status.risk))
        except Exception as e:
            result.add_error("risk", f"Risk validation failed: {e}")

        try:
            result.merge(self._validate_system(status.system))
        except Exception as e:
            result.add_error("system", f"System validation failed: {e}")

        return result

    def _validate_market(self, market: Any) -> ValidationResult:
        """Validate market data."""
        result = ValidationResult()

        if market is None:
            result.add_error("market", "Market data is None")
            return result

        # Validate prices
        if hasattr(market, "last_prices") and market.last_prices:
            for symbol, price in market.last_prices.items():
                price_result = self._validate_price(price, f"market.prices.{symbol}")
                result.merge(price_result)

        # Validate last_price_update timestamp
        if hasattr(market, "last_price_update"):
            if market.last_price_update is not None:
                if not isinstance(market.last_price_update, (int, float)):
                    result.add_warning(
                        "market.last_price_update",
                        f"Invalid timestamp type: {type(market.last_price_update)}",
                        market.last_price_update,
                    )
                elif market.last_price_update < 0:
                    result.add_warning(
                        "market.last_price_update",
                        "Negative timestamp",
                        market.last_price_update,
                    )

        return result

    def _validate_positions(self, positions: Any) -> ValidationResult:
        """Validate position data."""
        result = ValidationResult()

        if positions is None:
            result.add_error("positions", "Position data is None")
            return result

        # Validate individual positions
        if hasattr(positions, "positions") and positions.positions:
            for symbol, pos in positions.positions.items():
                if isinstance(pos, dict):
                    # Validate quantity
                    quantity = pos.get("quantity")
                    if quantity is not None:
                        quantity_result = self._validate_quantity(
                            quantity, f"positions.{symbol}.quantity"
                        )
                        result.merge(quantity_result)

                    # Validate entry price
                    entry_price = pos.get("entry_price")
                    if entry_price is not None:
                        price_result = self._validate_price(
                            entry_price, f"positions.{symbol}.entry_price"
                        )
                        result.merge(price_result)

                    # Validate side
                    side = pos.get("side", "")
                    if side and side not in ("LONG", "SHORT", "long", "short", ""):
                        result.add_warning(
                            f"positions.{symbol}.side",
                            f"Unexpected side value: {side}",
                            side,
                        )

        # Validate totals
        if hasattr(positions, "total_unrealized_pnl"):
            pnl_result = self._validate_decimal(
                positions.total_unrealized_pnl, "positions.total_unrealized_pnl"
            )
            result.merge(pnl_result)

        if hasattr(positions, "equity"):
            equity_result = self._validate_decimal(positions.equity, "positions.equity")
            result.merge(equity_result)

        return result

    def _validate_orders(self, orders: list[Any]) -> ValidationResult:
        """Validate order data."""
        result = ValidationResult()

        if orders is None:
            result.add_error("orders", "Order data is None")
            return result

        for i, order in enumerate(orders):
            # Validate required fields
            if not hasattr(order, "order_id") or not order.order_id:
                result.add_warning(f"orders[{i}].order_id", "Missing order_id")

            if not hasattr(order, "symbol") or not order.symbol:
                result.add_warning(f"orders[{i}].symbol", "Missing symbol")

            # Validate quantity
            if hasattr(order, "quantity"):
                quantity_result = self._validate_quantity(order.quantity, f"orders[{i}].quantity")
                result.merge(quantity_result)

            # Validate price
            if hasattr(order, "price") and order.price is not None:
                price_result = self._validate_price(order.price, f"orders[{i}].price")
                result.merge(price_result)

            # Validate side
            if hasattr(order, "side"):
                if order.side not in ("BUY", "SELL", "buy", "sell"):
                    result.add_warning(
                        f"orders[{i}].side",
                        f"Unexpected side: {order.side}",
                        order.side,
                    )

        return result

    def _validate_trades(self, trades: list[Any]) -> ValidationResult:
        """Validate trade data."""
        result = ValidationResult()

        if trades is None:
            result.add_error("trades", "Trade data is None")
            return result

        for i, trade in enumerate(trades):
            # Validate required fields
            if not hasattr(trade, "trade_id") or not trade.trade_id:
                result.add_warning(f"trades[{i}].trade_id", "Missing trade_id")

            if not hasattr(trade, "symbol") or not trade.symbol:
                result.add_warning(f"trades[{i}].symbol", "Missing symbol")

            # Validate quantity (must be positive)
            if hasattr(trade, "quantity"):
                quantity_result = self._validate_quantity(
                    trade.quantity, f"trades[{i}].quantity", allow_negative=False
                )
                result.merge(quantity_result)

            # Validate price (must be positive)
            if hasattr(trade, "price"):
                price_result = self._validate_price(trade.price, f"trades[{i}].price")
                result.merge(price_result)

            # Validate fee (should be non-negative)
            if hasattr(trade, "fee"):
                fee_result = self._validate_decimal(trade.fee, f"trades[{i}].fee")
                result.merge(fee_result)

        return result

    def _validate_account(self, account: Any) -> ValidationResult:
        """Validate account data."""
        result = ValidationResult()

        if account is None:
            result.add_error("account", "Account data is None")
            return result

        # Validate balances
        if hasattr(account, "balances"):
            for i, balance in enumerate(account.balances):
                if hasattr(balance, "total"):
                    total_result = self._validate_decimal(
                        balance.total, f"account.balances[{i}].total"
                    )
                    result.merge(total_result)

                if hasattr(balance, "available"):
                    avail_result = self._validate_decimal(
                        balance.available, f"account.balances[{i}].available"
                    )
                    result.merge(avail_result)

                # Check consistency: available should not exceed total
                if hasattr(balance, "total") and hasattr(balance, "available"):
                    try:
                        total = Decimal(str(balance.total))
                        available = Decimal(str(balance.available))
                        if available > total:
                            result.add_warning(
                                f"account.balances[{i}]",
                                f"Available ({available}) exceeds total ({total})",
                            )
                    except (InvalidOperation, ValueError):
                        pass  # Already caught by decimal validation

        return result

    def _validate_risk(self, risk: Any) -> ValidationResult:
        """Validate risk data."""
        result = ValidationResult()

        if risk is None:
            result.add_error("risk", "Risk data is None")
            return result

        # Validate leverage
        if hasattr(risk, "max_leverage"):
            if not isinstance(risk.max_leverage, (int, float)):
                result.add_warning(
                    "risk.max_leverage",
                    f"Invalid leverage type: {type(risk.max_leverage)}",
                )
            elif risk.max_leverage < 0 or risk.max_leverage > self.MAX_LEVERAGE:
                result.add_warning(
                    "risk.max_leverage",
                    f"Leverage out of range: {risk.max_leverage}",
                    risk.max_leverage,
                )

        # Validate daily loss limit percentage
        if hasattr(risk, "daily_loss_limit_pct"):
            if not isinstance(risk.daily_loss_limit_pct, (int, float)):
                result.add_warning(
                    "risk.daily_loss_limit_pct",
                    f"Invalid type: {type(risk.daily_loss_limit_pct)}",
                )
            elif risk.daily_loss_limit_pct < 0 or risk.daily_loss_limit_pct > 100:
                result.add_warning(
                    "risk.daily_loss_limit_pct",
                    f"Percentage out of range: {risk.daily_loss_limit_pct}",
                    risk.daily_loss_limit_pct,
                )

        # Validate current daily loss percentage
        if hasattr(risk, "current_daily_loss_pct"):
            if abs(risk.current_daily_loss_pct) > self.MAX_PNL_PERCENTAGE:
                result.add_warning(
                    "risk.current_daily_loss_pct",
                    f"Daily loss unusually high: {risk.current_daily_loss_pct}%",
                    risk.current_daily_loss_pct,
                )

        return result

    def _validate_system(self, system: Any) -> ValidationResult:
        """Validate system data."""
        result = ValidationResult()

        if system is None:
            result.add_error("system", "System data is None")
            return result

        # Validate API latency
        if hasattr(system, "api_latency"):
            if not isinstance(system.api_latency, (int, float)):
                result.add_warning(
                    "system.api_latency",
                    f"Invalid latency type: {type(system.api_latency)}",
                )
            elif system.api_latency < 0:
                result.add_warning(
                    "system.api_latency",
                    "Negative latency",
                    system.api_latency,
                )

        # Validate connection status
        if hasattr(system, "connection_status"):
            valid_statuses = ("CONNECTED", "DISCONNECTED", "CONNECTING", "UNKNOWN")
            if system.connection_status not in valid_statuses:
                result.add_warning(
                    "system.connection_status",
                    f"Unknown status: {system.connection_status}",
                    system.connection_status,
                )

        return result

    def _validate_price(self, value: Any, field_name: str) -> ValidationResult:
        """Validate a price value."""
        result = ValidationResult()

        try:
            price = Decimal(str(value)) if not isinstance(value, Decimal) else value

            if price < self.MIN_PRICE:
                result.add_error(field_name, f"Negative price: {price}", value)
            elif price > self.MAX_PRICE:
                result.add_warning(field_name, f"Price unusually high: {price}", value)
        except (InvalidOperation, ValueError, TypeError) as e:
            result.add_error(field_name, f"Invalid price format: {e}", value)

        return result

    def _validate_quantity(
        self, value: Any, field_name: str, allow_negative: bool = True
    ) -> ValidationResult:
        """Validate a quantity value."""
        result = ValidationResult()

        try:
            quantity_value = Decimal(str(value)) if not isinstance(value, Decimal) else value

            if not allow_negative and quantity_value < 0:
                result.add_error(field_name, f"Negative quantity: {quantity_value}", value)
            elif abs(quantity_value) > self.MAX_QUANTITY:
                result.add_warning(field_name, f"Quantity unusually large: {quantity_value}", value)
        except (InvalidOperation, ValueError, TypeError) as e:
            result.add_error(field_name, f"Invalid quantity format: {e}", value)

        return result

    def _validate_decimal(self, value: Any, field_name: str) -> ValidationResult:
        """Validate a general decimal value."""
        result = ValidationResult()

        if value is None:
            return result  # None is allowed for optional fields

        try:
            if not isinstance(value, Decimal):
                Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as e:
            result.add_error(field_name, f"Invalid decimal format: {e}", value)

        return result
