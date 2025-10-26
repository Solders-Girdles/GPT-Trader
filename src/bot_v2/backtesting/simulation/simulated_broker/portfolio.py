from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Dict, Iterable, List, Tuple

from bot_v2.backtesting.simulation.fee_calculator import FeeCalculator
from bot_v2.backtesting.simulation.funding_tracker import FundingPnLTracker
from bot_v2.backtesting.types import BacktestResult, FeeTier
from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    Position,
    Product,
)


class PortfolioManager:
    def __init__(
        self,
        initial_equity_usd: Decimal,
        fee_tier: FeeTier,
        enable_funding_pnl: bool,
    ) -> None:
        self._initial_equity = initial_equity_usd
        self._cash_balance = initial_equity_usd
        self._positions: Dict[str, Position] = {}
        self._products: Dict[str, Product] = {}

        self._fee_calculator = FeeCalculator(tier=fee_tier)
        self._funding_tracker = FundingPnLTracker() if enable_funding_pnl else None

        self._realized_pnl = Decimal("0")
        self._fees_paid = Decimal("0")
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0

        self._peak_equity = initial_equity_usd
        self._max_drawdown = Decimal("0")
        self._equity_history: List[Tuple[datetime, Decimal]] = []

    # ------------------------------------------------------------------
    def register_product(self, product: Product) -> None:
        self._products[product.symbol] = product

    def list_products(self, market: MarketType | None = None) -> List[Product]:
        products = list(self._products.values())
        if market:
            products = [p for p in products if p.market_type == market]
        return products

    def get_product(self, symbol: str) -> Product:
        if symbol not in self._products:
            raise KeyError(f"Product not found: {symbol}")
        return self._products[symbol]

    # ------------------------------------------------------------------
    def list_balances(self) -> List[Balance]:
        return [
            Balance(
                asset="USDC",
                total=self._cash_balance,
                available=self._cash_balance,
                hold=Decimal("0"),
            )
        ]

    def list_positions(self) -> List[Position]:
        return list(self._positions.values())

    # ------------------------------------------------------------------
    def has_sufficient_margin(self, required_notional: Decimal) -> bool:
        return self._cash_balance >= required_notional

    def update_marks(self, quotes: Iterable[Tuple[str, object]]) -> None:
        from bot_v2.features.brokerages.core.interfaces import Quote

        for symbol, quote_obj in quotes:
            if symbol not in self._positions:
                continue
            quote: Quote = quote_obj  # type: ignore[assignment]
            position = self._positions[symbol]
            mid_price = (quote.bid + quote.ask) / Decimal("2")
            position.mark_price = mid_price
            if position.side == "long":
                position.unrealized_pnl = (mid_price - position.entry_price) * position.quantity
            else:
                position.unrealized_pnl = (position.entry_price - mid_price) * abs(position.quantity)

    def accrue_funding(self, current_time: datetime) -> None:
        if not self._funding_tracker:
            return

        for symbol, position in self._positions.items():
            product = self._products.get(symbol)
            if not product or product.market_type != MarketType.PERPETUAL:
                continue

            funding_rate = product.funding_rate or Decimal("0.0001")
            funding = self._funding_tracker.accrue(
                symbol=symbol,
                position_size=position.quantity,
                mark_price=position.mark_price,
                funding_rate_8h=funding_rate,
                current_time=current_time,
            )
            if funding != Decimal("0"):
                self._cash_balance -= funding

    # ------------------------------------------------------------------
    def execute_fill(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal,
        is_maker: bool,
        current_time: datetime,
    ) -> None:
        notional = fill_quantity * fill_price
        fee = self._fee_calculator.calculate(notional, is_maker=is_maker)
        self._fees_paid += fee

        order.status = OrderStatus.FILLED
        order.filled_quantity = fill_quantity
        order.avg_fill_price = fill_price
        order.updated_at = current_time

        symbol = order.symbol
        if symbol not in self._positions:
            side = "long" if order.side == OrderSide.BUY else "short"
            qty = fill_quantity if side == "long" else -fill_quantity
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=qty,
                entry_price=fill_price,
                mark_price=fill_price,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side=side,
            )
        else:
            self._update_existing_position(symbol, order.side, fill_quantity, fill_price)

        if order.side == OrderSide.BUY:
            self._cash_balance -= (notional + fee)
        else:
            self._cash_balance += (notional - fee)

        self._total_trades += 1

    def _update_existing_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
    ) -> None:
        position = self._positions[symbol]
        current_qty = position.quantity
        entry_price = position.entry_price

        new_qty = current_qty + quantity if side == OrderSide.BUY else current_qty - quantity

        closing = (current_qty > 0 and new_qty < current_qty) or (current_qty < 0 and new_qty > current_qty)
        if closing:
            closed_qty = min(abs(quantity), abs(current_qty))
            if current_qty > 0:
                pnl = (price - entry_price) * closed_qty
            else:
                pnl = (entry_price - price) * closed_qty

            self._realized_pnl += pnl
            position.realized_pnl += pnl
            if pnl > 0:
                self._winning_trades += 1
            else:
                self._losing_trades += 1

        if abs(new_qty) < Decimal("0.00000001"):
            del self._positions[symbol]
            return

        position.quantity = new_qty
        if (current_qty > 0 and new_qty > current_qty) or (current_qty < 0 and new_qty < current_qty):
            total_cost = (abs(current_qty) * entry_price) + (quantity * price)
            total_qty = abs(new_qty)
            if total_qty > 0:
                position.entry_price = total_cost / total_qty
        position.side = "long" if new_qty > 0 else "short"

    # ------------------------------------------------------------------
    def get_equity(self) -> Decimal:
        equity = self._cash_balance
        for position in self._positions.values():
            equity += position.unrealized_pnl
        return equity

    def get_margin_used(self) -> Decimal:
        margin = Decimal("0")
        for position in self._positions.values():
            margin += abs(position.quantity) * position.mark_price
        return margin

    def get_margin_available(self) -> Decimal:
        return self.get_equity() - self.get_margin_used()

    def record_equity(self, timestamp: datetime) -> None:
        equity = self.get_equity()
        self._equity_history.append((timestamp, equity))
        if equity > self._peak_equity:
            self._peak_equity = equity
        else:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown

    # ------------------------------------------------------------------
    def generate_report(self) -> BacktestResult:
        if not self._equity_history:
            raise ValueError("No equity history available")

        start_time = self._equity_history[0][0]
        end_time = self._equity_history[-1][0]
        duration = (end_time - start_time).days

        final_equity = self.get_equity()
        total_return_usd = final_equity - self._initial_equity
        total_return_pct = (total_return_usd / self._initial_equity) * Decimal("100")

        win_rate = (
            Decimal(self._winning_trades) / Decimal(self._total_trades) * Decimal("100")
            if self._total_trades > 0
            else Decimal("0")
        )

        funding_pnl = (
            self._funding_tracker.get_total_funding_pnl()
            if self._funding_tracker
            else Decimal("0")
        )

        unrealized_total = sum(p.unrealized_pnl for p in self._positions.values())

        return BacktestResult(
            start_date=start_time,
            end_date=end_time,
            duration_days=duration,
            initial_equity=self._initial_equity,
            final_equity=final_equity,
            total_return=total_return_pct,
            total_return_usd=total_return_usd,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=unrealized_total,
            funding_pnl=funding_pnl,
            fees_paid=self._fees_paid,
            total_trades=self._total_trades,
            winning_trades=self._winning_trades,
            losing_trades=self._losing_trades,
            win_rate=win_rate,
            max_drawdown=self._max_drawdown * Decimal("100"),
            max_drawdown_usd=self._peak_equity - (self._peak_equity * (Decimal("1") - self._max_drawdown)),
        )

    # ------------------------------------------------------------------
    @property
    def products(self) -> Dict[str, Product]:
        return self._products

    @property
    def funding_tracker(self) -> FundingPnLTracker | None:
        return self._funding_tracker

    @property
    def fee_calculator(self) -> FeeCalculator:
        return self._fee_calculator

    @property
    def cash_balance(self) -> Decimal:
        return self._cash_balance


__all__ = ["PortfolioManager"]
