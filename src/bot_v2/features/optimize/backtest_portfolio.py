"""
Portfolio state simulation for production-parity backtesting.

Maintains portfolio state (positions, cash, equity) and simulates
trade execution with realistic fees and slippage.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.core.interfaces import Product
from bot_v2.features.live_trade.strategies.decisions import Action, Decision
from bot_v2.utilities.logging_patterns import get_logger

from .types_v2 import BacktestConfig, ExecutionResult

logger = get_logger(__name__, component="optimize")


class BacktestPortfolio:
    """Simulates portfolio state during backtesting."""

    def __init__(
        self,
        *,
        initial_capital: Decimal,
        commission_rate: Decimal = Decimal("0.001"),
        slippage_rate: Decimal = Decimal("0.0005"),
    ):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting cash
            commission_rate: Commission as fraction (0.001 = 0.1% = 10 bps)
            slippage_rate: Slippage as fraction (0.0005 = 0.05% = 5 bps)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        # Portfolio state
        self.cash = initial_capital
        self.positions: dict[str, dict[str, Any]] = {}  # symbol -> position_state
        self.equity_history: list[tuple[datetime, Decimal]] = []

        # Trade tracking
        self.trade_count = 0
        self.total_commission = Decimal("0")

    def get_position_state(self, symbol: str) -> dict[str, Any] | None:
        """
        Get current position state for a symbol.

        Returns:
            Position dict with keys: quantity, side, entry, unrealized_pnl
            None if no position
        """
        return self.positions.get(symbol)

    def get_equity(self, current_prices: dict[str, Decimal] | None = None) -> Decimal:
        """
        Calculate current equity.

        Args:
            current_prices: Current mark prices for open positions

        Returns:
            Total equity (cash + unrealized position value)
        """
        equity = self.cash

        if current_prices:
            for symbol, position in self.positions.items():
                if symbol not in current_prices:
                    logger.warning(
                        "Missing price for position | symbol=%s, assuming entry price", symbol
                    )
                    current_price = position["entry"]
                else:
                    current_price = current_prices[symbol]

                # Calculate position value
                quantity = position["quantity"]
                side = position["side"]
                entry_price = position["entry"]

                if side == "long":
                    position_value = quantity * current_price
                    cost_basis = quantity * entry_price
                    unrealized_pnl = position_value - cost_basis
                else:  # short
                    # For shorts: we received cash at entry, owe shares at current price
                    proceeds_at_entry = quantity * entry_price
                    cost_to_close = quantity * current_price
                    unrealized_pnl = proceeds_at_entry - cost_to_close

                equity += unrealized_pnl

        return equity

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
        Execute a strategy decision and update portfolio state.

        Args:
            decision: Strategy decision
            symbol: Trading symbol
            current_price: Current mark price
            product: Product metadata
            timestamp: Current timestamp

        Returns:
            ExecutionResult with fill details or rejection
        """
        if decision.action == Action.HOLD:
            return ExecutionResult(filled=False, rejection_reason="HOLD action")

        # Check if decision was already rejected by filters/guards
        if decision.filter_rejected or decision.guard_rejected:
            return ExecutionResult(
                filled=False,
                rejection_reason=decision.rejection_type or "Filter/guard rejection",
            )

        # Execute based on action
        if decision.action == Action.BUY:
            return self._execute_buy(
                symbol=symbol,
                decision=decision,
                current_price=current_price,
                product=product,
                timestamp=timestamp,
            )
        elif decision.action == Action.SELL:
            return self._execute_sell(
                symbol=symbol,
                decision=decision,
                current_price=current_price,
                product=product,
                timestamp=timestamp,
            )
        elif decision.action == Action.CLOSE:
            return self._execute_close(
                symbol=symbol,
                decision=decision,
                current_price=current_price,
                timestamp=timestamp,
            )
        else:
            return ExecutionResult(filled=False, rejection_reason=f"Unknown action: {decision.action}")

    def _execute_buy(
        self,
        *,
        symbol: str,
        decision: Decision,
        current_price: Decimal,
        product: Product,
        timestamp: datetime,
    ) -> ExecutionResult:
        """Execute a buy order."""
        # Calculate quantity from target notional
        if decision.quantity is not None:
            quantity = decision.quantity
        elif decision.target_notional is not None:
            # Apply slippage to execution price (buy = pay higher)
            exec_price = current_price * (Decimal("1") + self.slippage_rate)
            quantity = decision.target_notional / exec_price
        else:
            return ExecutionResult(filled=False, rejection_reason="No quantity or notional specified")

        # Round to product step size
        quantity = self._round_quantity(quantity, product)

        if quantity < product.min_size:
            return ExecutionResult(
                filled=False, rejection_reason=f"Quantity {quantity} below min_size {product.min_size}"
            )

        # Apply slippage
        fill_price = current_price * (Decimal("1") + self.slippage_rate)

        # Calculate costs
        notional = quantity * fill_price
        commission = notional * self.commission_rate
        total_cost = notional + commission

        # Check sufficient cash
        if total_cost > self.cash:
            return ExecutionResult(
                filled=False,
                rejection_reason=f"Insufficient cash: need {total_cost}, have {self.cash}",
            )

        # Execute
        self.cash -= total_cost
        self.positions[symbol] = {
            "quantity": quantity,
            "side": "long",
            "entry": fill_price,
            "entry_time": timestamp,
        }
        self.trade_count += 1
        self.total_commission += commission

        logger.debug(
            "BUY executed | symbol=%s | qty=%s | price=%s | commission=%s",
            symbol,
            quantity,
            fill_price,
            commission,
        )

        return ExecutionResult(
            filled=True,
            fill_price=fill_price,
            filled_quantity=quantity,
            commission=commission,
            slippage=fill_price - current_price,
        )

    def _execute_sell(
        self,
        *,
        symbol: str,
        decision: Decision,
        current_price: Decimal,
        product: Product,
        timestamp: datetime,
    ) -> ExecutionResult:
        """Execute a sell (short) order."""
        # Calculate quantity from target notional
        if decision.quantity is not None:
            quantity = decision.quantity
        elif decision.target_notional is not None:
            # Apply slippage to execution price (sell = receive lower)
            exec_price = current_price * (Decimal("1") - self.slippage_rate)
            quantity = decision.target_notional / exec_price
        else:
            return ExecutionResult(filled=False, rejection_reason="No quantity or notional specified")

        # Round to product step size
        quantity = self._round_quantity(quantity, product)

        if quantity < product.min_size:
            return ExecutionResult(
                filled=False, rejection_reason=f"Quantity {quantity} below min_size {product.min_size}"
            )

        # Apply slippage (sell = receive lower price)
        fill_price = current_price * (Decimal("1") - self.slippage_rate)

        # Calculate proceeds
        notional = quantity * fill_price
        commission = notional * self.commission_rate
        net_proceeds = notional - commission

        # Execute (short position)
        self.cash += net_proceeds
        self.positions[symbol] = {
            "quantity": quantity,
            "side": "short",
            "entry": fill_price,
            "entry_time": timestamp,
        }
        self.trade_count += 1
        self.total_commission += commission

        logger.debug(
            "SELL executed | symbol=%s | qty=%s | price=%s | commission=%s",
            symbol,
            quantity,
            fill_price,
            commission,
        )

        return ExecutionResult(
            filled=True,
            fill_price=fill_price,
            filled_quantity=quantity,
            commission=commission,
            slippage=current_price - fill_price,
        )

    def _execute_close(
        self,
        *,
        symbol: str,
        decision: Decision,
        current_price: Decimal,
        timestamp: datetime,
    ) -> ExecutionResult:
        """Execute a close order (close existing position)."""
        position = self.positions.get(symbol)
        if not position:
            return ExecutionResult(filled=False, rejection_reason="No position to close")

        quantity = position["quantity"]
        side = position["side"]
        entry_price = position["entry"]

        # Close logic depends on position side
        if side == "long":
            # Close long = sell
            fill_price = current_price * (Decimal("1") - self.slippage_rate)
            proceeds = quantity * fill_price
            commission = proceeds * self.commission_rate
            net_proceeds = proceeds - commission

            self.cash += net_proceeds

            # Calculate realized P&L
            cost_basis = quantity * entry_price
            realized_pnl = proceeds - cost_basis - commission

        else:  # short
            # Close short = buy to cover
            fill_price = current_price * (Decimal("1") + self.slippage_rate)
            cost_to_cover = quantity * fill_price
            commission = cost_to_cover * self.commission_rate
            total_cost = cost_to_cover + commission

            self.cash -= total_cost

            # Calculate realized P&L
            proceeds_at_entry = quantity * entry_price
            realized_pnl = proceeds_at_entry - cost_to_cover - commission

        # Remove position
        del self.positions[symbol]
        self.trade_count += 1
        self.total_commission += commission

        logger.debug(
            "CLOSE executed | symbol=%s | side=%s | qty=%s | entry=%s | exit=%s | pnl=%s",
            symbol,
            side,
            quantity,
            entry_price,
            fill_price,
            realized_pnl,
        )

        return ExecutionResult(
            filled=True,
            fill_price=fill_price,
            filled_quantity=quantity,
            commission=commission,
            slippage=abs(fill_price - current_price),
        )

    def record_equity(self, timestamp: datetime, current_prices: dict[str, Decimal]) -> None:
        """Record equity snapshot for equity curve."""
        equity = self.get_equity(current_prices)
        self.equity_history.append((timestamp, equity))

    def get_equity_curve(self) -> list[tuple[datetime, Decimal]]:
        """Get full equity curve."""
        return self.equity_history

    def get_stats(self) -> dict[str, Any]:
        """Get portfolio statistics."""
        current_equity = self.get_equity()
        total_return = (current_equity - self.initial_capital) / self.initial_capital

        return {
            "initial_capital": self.initial_capital,
            "current_equity": current_equity,
            "cash": self.cash,
            "total_return": float(total_return),
            "trade_count": self.trade_count,
            "total_commission": self.total_commission,
            "open_positions": len(self.positions),
        }

    def _round_quantity(self, quantity: Decimal, product: Product) -> Decimal:
        """Round quantity to product step size."""
        if product.step_size == Decimal("0"):
            return quantity

        # Round down to nearest step
        steps = quantity / product.step_size
        rounded_steps = int(steps)
        return rounded_steps * product.step_size


__all__ = ["BacktestPortfolio"]
