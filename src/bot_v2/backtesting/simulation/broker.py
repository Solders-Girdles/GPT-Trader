"""Simulated broker for backtesting that implements IBrokerage interface."""

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

from bot_v2.backtesting.simulation.fee_calculator import FeeCalculator
from bot_v2.backtesting.simulation.fill_model import OrderFillModel
from bot_v2.backtesting.simulation.funding_tracker import FundingPnLTracker
from bot_v2.backtesting.types import BacktestResult, FeeTier
from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    Candle,
    InsufficientFunds,
    InvalidRequestError,
    MarketType,
    NotFoundError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    Quote,
    TimeInForce,
)


class SimulatedBroker:
    """
    Simulated broker that implements IBrokerage for backtesting.

    This broker maintains an in-memory state of balances, positions, and orders,
    and simulates realistic order fills with fees, slippage, and funding PnL.
    """

    def __init__(
        self,
        initial_equity_usd: Decimal = Decimal("100000"),
        fee_tier: FeeTier = FeeTier.TIER_2,
        slippage_bps: dict[str, Decimal] | None = None,
        spread_impact_pct: Decimal = Decimal("0.5"),
        enable_funding_pnl: bool = True,
    ):
        """
        Initialize simulated broker.

        Args:
            initial_equity_usd: Starting capital in USD
            fee_tier: Coinbase Advanced Trade fee tier
            slippage_bps: Per-symbol slippage (basis points)
            spread_impact_pct: Fraction of spread to apply to market orders
            enable_funding_pnl: Track funding PnL for perps
        """
        # Capital
        self._initial_equity = initial_equity_usd
        self._cash_balance = initial_equity_usd  # Available cash (USDC)

        # Positions {symbol: Position}
        self._positions: dict[str, Position] = {}

        # Orders {order_id: Order}
        self._orders: dict[str, Order] = {}
        self._open_orders: dict[str, Order] = {}  # Subset of orders that are open

        # Products {symbol: Product}
        self._products: dict[str, Product] = {}

        # Current market data {symbol: (candle, bid, ask)}
        self._current_bar: dict[str, Candle] = {}
        self._current_quotes: dict[str, Quote] = {}
        self._next_bar: dict[str, Candle] = {}  # For market order fills

        # Simulation components
        self._fee_calculator = FeeCalculator(tier=fee_tier)
        self._fill_model = OrderFillModel(
            slippage_bps=slippage_bps,
            spread_impact_pct=spread_impact_pct,
        )
        self._funding_tracker = FundingPnLTracker() if enable_funding_pnl else None

        # Performance tracking
        self._realized_pnl = Decimal("0")
        self._fees_paid = Decimal("0")
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0

        # Equity tracking for drawdown calculation
        self._peak_equity = initial_equity_usd
        self._max_drawdown = Decimal("0")
        self._equity_history: list[tuple[datetime, Decimal]] = []

        # Current time
        self._current_time = datetime.now(tz=timezone.utc)

    def update_market_data(
        self,
        current_time: datetime,
        bars: dict[str, Candle],
        quotes: dict[str, Quote] | None = None,
        next_bars: dict[str, Candle] | None = None,
    ) -> None:
        """
        Update current market data for simulation.

        This should be called by the bar runner before each strategy cycle.

        Args:
            current_time: Current simulation time
            bars: Current candles {symbol: Candle}
            quotes: Current quotes {symbol: Quote} (optional, derived from bars if not provided)
            next_bars: Next candles for market order fills (optional)
        """
        self._current_time = current_time
        self._current_bar = bars
        self._next_bar = next_bars or {}

        # Update quotes (use provided or derive from bars)
        if quotes:
            self._current_quotes = quotes
        else:
            self._current_quotes = {
                symbol: Quote(
                    symbol=symbol,
                    bid=bar.close * Decimal("0.9995"),  # Estimate bid
                    ask=bar.close * Decimal("1.0005"),  # Estimate ask
                    last=bar.close,
                    ts=bar.ts,
                )
                for symbol, bar in bars.items()
            }

        # Update mark prices for positions
        self._update_position_marks()

        # Try to fill pending limit/stop orders
        self._process_pending_orders()

        # Accrue funding PnL if enabled
        if self._funding_tracker:
            self._accrue_funding_pnl()

        # Track equity
        equity = self.get_equity()
        self._equity_history.append((current_time, equity))

        # Update peak and drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity
        else:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown

    # IBrokerage interface implementation

    def connect(self) -> bool:
        """Connect to broker (no-op for simulation)."""
        return True

    def disconnect(self) -> None:
        """Disconnect from broker (no-op for simulation)."""
        pass

    def validate_connection(self) -> bool:
        """Validate broker connection (always True for simulation)."""
        return True

    def get_account_id(self) -> str:
        """Get account ID."""
        return "SIMULATED_ACCOUNT"

    def list_balances(self) -> list[Balance]:
        """Get current balances."""
        # For simplicity, we only track USDC balance
        return [
            Balance(
                asset="USDC",
                total=self._cash_balance,
                available=self._cash_balance,
                hold=Decimal("0"),
            )
        ]

    def get_account_info(self) -> dict:
        """Get account information."""
        return {
            "equity": self.get_equity(),
            "cash": self._cash_balance,
            "margin_used": self.get_margin_used(),
            "margin_available": self.get_margin_available(),
        }

    def list_products(self, market: MarketType | None = None) -> list[Product]:
        """List available products."""
        products = list(self._products.values())
        if market:
            products = [p for p in products if p.market_type == market]
        return products

    def get_product(self, symbol: str) -> Product:
        """Get product by symbol."""
        if symbol not in self._products:
            raise NotFoundError(f"Product not found: {symbol}")
        return self._products[symbol]

    def register_product(self, product: Product) -> None:
        """Register a product for trading (simulation-specific method)."""
        self._products[product.symbol] = product

    def get_quote(self, symbol: str) -> Quote:
        """Get current quote for symbol."""
        if symbol not in self._current_quotes:
            raise NotFoundError(f"Quote not available for {symbol}")
        return self._current_quotes[symbol]

    def get_candles(
        self,
        symbol: str,
        granularity: str,
        limit: int = 300,
    ) -> list[Candle]:
        """Get historical candles (limited to current bar in simulation)."""
        if symbol in self._current_bar:
            return [self._current_bar[symbol]]
        return []

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: str,
        limit_price: str | None = None,
        stop_price: str | None = None,
        time_in_force: str | None = None,
        post_only: bool = False,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> Order:
        """Place an order."""
        # Validate inputs
        try:
            qty = Decimal(quantity)
            side_enum = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
            type_enum = OrderType[order_type.upper()]
            tif = TimeInForce[time_in_force.upper()] if time_in_force else TimeInForce.GTC

            limit_px = Decimal(limit_price) if limit_price else None
            stop_px = Decimal(stop_price) if stop_price else None
        except (ValueError, KeyError) as e:
            raise InvalidRequestError(f"Invalid order parameters: {e}") from e

        # Validate product exists
        if symbol not in self._products:
            raise InvalidRequestError(f"Unknown product: {symbol}")

        product = self._products[symbol]

        # Validate order size
        if qty < product.min_size:
            raise InvalidRequestError(
                f"Order size {qty} below minimum {product.min_size}"
            )

        # Check if we have enough margin/cash
        notional = qty * (limit_px or self._current_quotes[symbol].last)
        if side_enum == OrderSide.BUY and not self._has_sufficient_margin(notional):
            raise InsufficientFunds(f"Insufficient funds for order (need {notional})")

        # Create order
        order_id = str(uuid.uuid4())
        order = Order(
            id=order_id,
            client_id=client_order_id,
            symbol=symbol,
            side=side_enum,
            type=type_enum,
            quantity=qty,
            price=limit_px,
            stop_price=stop_px,
            tif=tif,
            status=OrderStatus.PENDING,
            submitted_at=self._current_time,
            updated_at=self._current_time,
        )

        # Store order
        self._orders[order_id] = order

        # Try to fill immediately if market order
        if type_enum == OrderType.MARKET:
            self._fill_market_order(order)
        else:
            # Add to open orders for later processing
            self._open_orders[order_id] = order
            order.status = OrderStatus.SUBMITTED

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self._orders:
            raise NotFoundError(f"Order not found: {order_id}")

        order = self._orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = self._current_time
        self._open_orders.pop(order_id, None)
        return True

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    def list_orders(
        self,
        status: str | None = None,
        symbol: str | None = None,
    ) -> list[Order]:
        """List orders with optional filters."""
        orders = list(self._orders.values())

        if status:
            status_enum = OrderStatus[status.upper()]
            orders = [o for o in orders if o.status == status_enum]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def list_positions(self) -> list[Position]:
        """List current positions."""
        return list(self._positions.values())

    def list_fills(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List fills (not fully implemented for simulation)."""
        # For simulation, we could track fills separately
        # For now, return empty list
        return []

    # Internal methods

    def _fill_market_order(self, order: Order) -> None:
        """Fill a market order immediately."""
        symbol = order.symbol
        quote = self._current_quotes[symbol]
        current_bar = self._current_bar[symbol]
        next_bar = self._next_bar.get(symbol)

        fill_result = self._fill_model.fill_market_order(
            order=order,
            current_bar=current_bar,
            best_bid=quote.bid,
            best_ask=quote.ask,
            next_bar=next_bar,
        )

        if fill_result.filled:
            self._execute_fill(
                order=order,
                fill_price=fill_result.fill_price,
                fill_quantity=fill_result.fill_quantity,
                is_maker=fill_result.is_maker,
            )

    def _process_pending_orders(self) -> None:
        """Process pending limit/stop orders."""
        filled_orders = []

        for order_id, order in self._open_orders.items():
            symbol = order.symbol
            if symbol not in self._current_bar:
                continue

            current_bar = self._current_bar[symbol]
            quote = self._current_quotes[symbol]

            if order.type == OrderType.LIMIT:
                fill_result = self._fill_model.try_fill_limit_order(
                    order=order,
                    current_bar=current_bar,
                    best_bid=quote.bid,
                    best_ask=quote.ask,
                )
            elif order.type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                next_bar = self._next_bar.get(symbol)
                fill_result = self._fill_model.try_fill_stop_order(
                    order=order,
                    current_bar=current_bar,
                    best_bid=quote.bid,
                    best_ask=quote.ask,
                    next_bar=next_bar,
                )
            else:
                continue

            if fill_result.filled:
                self._execute_fill(
                    order=order,
                    fill_price=fill_result.fill_price,
                    fill_quantity=fill_result.fill_quantity,
                    is_maker=fill_result.is_maker,
                )
                filled_orders.append(order_id)

        # Remove filled orders from open orders
        for order_id in filled_orders:
            self._open_orders.pop(order_id, None)

    def _execute_fill(
        self,
        order: Order,
        fill_price: Decimal,
        fill_quantity: Decimal,
        is_maker: bool,
    ) -> None:
        """Execute a fill and update positions."""
        symbol = order.symbol
        notional = fill_quantity * fill_price

        # Calculate fee
        fee = self._fee_calculator.calculate(notional, is_maker=is_maker)
        self._fees_paid += fee

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = fill_quantity
        order.avg_fill_price = fill_price
        order.updated_at = self._current_time

        # Update position
        if symbol not in self._positions:
            # Open new position
            side: Literal["long", "short"] = "long" if order.side == OrderSide.BUY else "short"
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
            # Update existing position
            self._update_position(symbol, order.side, fill_quantity, fill_price)

        # Update cash balance
        if order.side == OrderSide.BUY:
            self._cash_balance -= (notional + fee)
        else:
            self._cash_balance += (notional - fee)

        # Track trade statistics
        self._total_trades += 1

    def _update_position(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
    ) -> None:
        """Update an existing position."""
        position = self._positions[symbol]
        current_qty = position.quantity
        current_entry = position.entry_price

        if side == OrderSide.BUY:
            new_qty = current_qty + quantity
        else:
            new_qty = current_qty - quantity

        # Check if closing or reducing position
        if (current_qty > 0 and new_qty < current_qty) or (current_qty < 0 and new_qty > current_qty):
            # Closing or reducing - realize PnL
            closed_qty = min(abs(quantity), abs(current_qty))
            if current_qty > 0:  # Was long
                pnl = (price - current_entry) * closed_qty
            else:  # Was short
                pnl = (current_entry - price) * closed_qty

            self._realized_pnl += pnl
            position.realized_pnl += pnl

            if pnl > 0:
                self._winning_trades += 1
            else:
                self._losing_trades += 1

        # Update position quantity
        if abs(new_qty) < Decimal("0.00000001"):  # Essentially zero
            # Position closed
            del self._positions[symbol]
        else:
            position.quantity = new_qty
            # Update entry price (weighted average for additions)
            if (current_qty > 0 and new_qty > current_qty) or (current_qty < 0 and new_qty < current_qty):
                # Adding to position
                total_cost = (abs(current_qty) * current_entry) + (quantity * price)
                total_qty = abs(new_qty)
                position.entry_price = total_cost / total_qty

            position.side = "long" if new_qty > 0 else "short"

    def _update_position_marks(self) -> None:
        """Update mark prices for all positions."""
        for symbol, position in self._positions.items():
            if symbol in self._current_quotes:
                quote = self._current_quotes[symbol]
                mid_price = (quote.bid + quote.ask) / Decimal("2")
                position.mark_price = mid_price

                # Calculate unrealized PnL
                if position.side == "long":
                    position.unrealized_pnl = (mid_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - mid_price) * abs(position.quantity)

    def _accrue_funding_pnl(self) -> None:
        """Accrue funding PnL for perpetual positions."""
        if not self._funding_tracker:
            return

        for symbol, position in self._positions.items():
            product = self._products.get(symbol)
            if not product or product.market_type != MarketType.PERPETUAL:
                continue

            funding_rate = product.funding_rate or Decimal("0.0001")  # Default 0.01%
            funding = self._funding_tracker.accrue(
                symbol=symbol,
                position_size=position.quantity,
                mark_price=position.mark_price,
                funding_rate_8h=funding_rate,
                current_time=self._current_time,
            )

            # Deduct funding from cash
            if funding != Decimal("0"):
                self._cash_balance -= funding

    def _has_sufficient_margin(self, required_notional: Decimal) -> bool:
        """Check if there's sufficient margin for a new position."""
        # Simplified: just check cash balance
        return self._cash_balance >= required_notional

    def get_equity(self) -> Decimal:
        """Calculate total equity (cash + unrealized PnL)."""
        equity = self._cash_balance
        for position in self._positions.values():
            equity += position.unrealized_pnl
        return equity

    def get_margin_used(self) -> Decimal:
        """Calculate total margin used by positions."""
        margin = Decimal("0")
        for position in self._positions.values():
            notional = abs(position.quantity) * position.mark_price
            margin += notional
        return margin

    def get_margin_available(self) -> Decimal:
        """Calculate available margin."""
        return self.get_equity() - self.get_margin_used()

    def generate_report(self) -> BacktestResult:
        """Generate backtest performance report."""
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

        return BacktestResult(
            start_date=start_time,
            end_date=end_time,
            duration_days=duration,
            initial_equity=self._initial_equity,
            final_equity=final_equity,
            total_return=total_return_pct,
            total_return_usd=total_return_usd,
            realized_pnl=self._realized_pnl,
            unrealized_pnl=sum(p.unrealized_pnl for p in self._positions.values()),
            funding_pnl=funding_pnl,
            fees_paid=self._fees_paid,
            total_trades=self._total_trades,
            winning_trades=self._winning_trades,
            losing_trades=self._losing_trades,
            win_rate=win_rate,
            max_drawdown=self._max_drawdown * Decimal("100"),  # Convert to %
            max_drawdown_usd=self._peak_equity - (self._peak_equity * (Decimal("1") - self._max_drawdown)),
        )
