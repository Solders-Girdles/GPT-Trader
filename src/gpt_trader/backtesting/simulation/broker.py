"""
Simulated broker for backtesting with full BrokerProtocol support.

This broker simulates realistic order execution, position management,
and market data for backtesting trading strategies.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from gpt_trader.backtesting.simulation.fee_calculator import FeeCalculator
from gpt_trader.backtesting.simulation.fill_model import FillResult, OrderFillModel
from gpt_trader.backtesting.simulation.funding_tracker import FundingPnLTracker
from gpt_trader.backtesting.types import CompletedTrade, FeeTier, SimulationConfig
from gpt_trader.features.brokerages.core.interfaces import (
    Balance,
    Candle,
    InsufficientFunds,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    Quote,
)


class SimulatedBroker:
    """
    Full-featured simulated broker implementing BrokerProtocol.

    Provides realistic simulation of:
    - Order execution with slippage and fees
    - Position tracking with PnL calculation
    - Market data (quotes, tickers, candles)
    - Balance management
    - Extended protocol methods for risk management

    Zero Logic Drift: The same strategy code can run against this broker
    or CoinbaseRestService without modification.
    """

    def __init__(
        self,
        initial_equity_usd: Decimal = Decimal("100000"),
        fee_tier: FeeTier = FeeTier.TIER_2,
        config: SimulationConfig | None = None,
    ):
        """
        Initialize simulated broker.

        Args:
            initial_equity_usd: Starting capital in USD
            fee_tier: Coinbase fee tier for fee calculation
            config: Optional simulation configuration for advanced settings
        """
        self._initial_equity = initial_equity_usd
        self.fee_tier = fee_tier
        self.config = config

        # Account state
        self.balances: dict[str, Balance] = {
            "USDC": Balance(
                asset="USDC",
                total=initial_equity_usd,
                available=initial_equity_usd,
                hold=Decimal("0"),
            )
        }
        self.positions: dict[str, Position] = {}
        self._position_history: list[dict[str, Any]] = []

        # Order management
        self._open_orders: dict[str, Order] = {}
        self._filled_orders: dict[str, Order] = {}
        self._cancelled_orders: dict[str, Order] = {}
        self._order_history: list[Order] = []

        # Market state (updated by bar runner)
        self._current_bar: dict[str, Candle] = {}
        self._current_quote: dict[str, Quote] = {}
        self._candle_history: dict[str, list[Candle]] = {}

        # Product registry
        self.products: dict[str, Product] = {}

        # Simulation components
        self._fill_model = OrderFillModel(
            slippage_bps=config.slippage_bps if config else None,
            spread_impact_pct=config.spread_impact_pct if config else Decimal("0.5"),
            limit_volume_threshold=(
                config.limit_fill_volume_threshold if config else Decimal("2.0")
            ),
        )
        self._fee_calculator = FeeCalculator(tier=fee_tier)
        self._funding_tracker = FundingPnLTracker(
            accrual_interval_hours=config.funding_accrual_hours if config else 1,
            settlement_interval_hours=config.funding_settlement_hours if config else 8,
        )

        # Equity tracking
        self._equity_curve: list[tuple[datetime, Decimal]] = []
        self._peak_equity = initial_equity_usd
        self._max_drawdown = Decimal("0")
        self._max_drawdown_usd = Decimal("0")

        # Statistics
        self._total_fees_paid = Decimal("0")
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0
        self._total_slippage_bps = Decimal("0")

        # Trade history tracking
        self._completed_trades: list[CompletedTrade] = []
        self._position_entry_times: dict[str, datetime] = {}
        self._position_entry_fees: dict[str, Decimal] = {}

        # Connection state
        self.connected = False
        self._simulation_time: datetime | None = None

    # =========================================================================
    # Connection Methods
    # =========================================================================

    def connect(self) -> bool:
        """Establish connection (always succeeds for simulation)."""
        self.connected = True
        return True

    def validate_connection(self) -> bool:
        """Check if broker is connected."""
        return self.connected

    def get_account_id(self) -> str:
        """Get simulated account ID."""
        return "SIMULATED_ACCOUNT"

    def disconnect(self) -> None:
        """Disconnect broker."""
        self.connected = False

    # =========================================================================
    # Product Registry
    # =========================================================================

    def register_product(self, product: Product) -> None:
        """Register a product for trading."""
        self.products[product.symbol] = product

    def get_product(self, symbol: str) -> Product | None:
        """Get product metadata for a symbol."""
        return self.products.get(symbol)

    # =========================================================================
    # Balance & Account Methods
    # =========================================================================

    def list_balances(self) -> list[Balance]:
        """List all account balances."""
        return list(self.balances.values())

    @property
    def equity(self) -> Decimal:
        """Get total account equity including unrealized PnL."""
        return self.get_equity()

    def get_equity(self) -> Decimal:
        """Get total account equity including unrealized PnL."""
        cash = self.balances.get("USDC", Balance("USDC", Decimal("0"), Decimal("0")))
        unrealized_pnl = sum((p.unrealized_pnl for p in self.positions.values()), Decimal("0"))
        return cash.total + unrealized_pnl

    def get_account_info(self) -> dict[str, Decimal]:
        """Get account summary information."""
        equity = self.get_equity()
        return {
            "cash": self.balances.get(
                "USDC", Balance("USDC", Decimal("0"), Decimal("0"))
            ).available,
            "equity": equity,
            "unrealized_pnl": sum(
                (p.unrealized_pnl for p in self.positions.values()), Decimal("0")
            ),
            "realized_pnl": sum((p.realized_pnl for p in self.positions.values()), Decimal("0")),
            "margin_used": self._calculate_margin_used(),
        }

    def _calculate_margin_used(self) -> Decimal:
        """Calculate total margin used by open positions."""
        total_margin = Decimal("0")
        for pos in self.positions.values():
            notional = abs(pos.quantity) * pos.mark_price
            leverage = pos.leverage or 1
            total_margin += notional / Decimal(leverage)
        return total_margin

    # =========================================================================
    # Position Methods
    # =========================================================================

    def list_positions(self) -> list[Position]:
        """List all current positions."""
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol."""
        return self.positions.get(symbol)

    # =========================================================================
    # Market Data Methods
    # =========================================================================

    def get_quote(self, symbol: str) -> Quote | None:
        """Get current quote for a symbol."""
        return self._current_quote.get(symbol)

    def get_ticker(self, product_id: str) -> dict[str, Any]:
        """Get ticker data for a product."""
        quote = self._current_quote.get(product_id)
        bar = self._current_bar.get(product_id)

        if quote:
            return {
                "price": str(quote.last),
                "bid": str(quote.bid),
                "ask": str(quote.ask),
                "volume": str(bar.volume) if bar else "0",
                "time": quote.ts.isoformat() if quote.ts else None,
            }

        if bar:
            return {
                "price": str(bar.close),
                "bid": str(bar.close),
                "ask": str(bar.close),
                "volume": str(bar.volume),
                "time": bar.ts.isoformat() if bar.ts else None,
            }

        return {"price": "0", "bid": "0", "ask": "0", "volume": "0"}

    def get_candles(self, symbol: str, **kwargs: Any) -> list[Candle]:
        """Get historical candle data for a symbol."""
        history = self._candle_history.get(symbol, [])
        limit = kwargs.get("limit", 300)
        return history[-limit:] if len(history) > limit else history

    # =========================================================================
    # Order Methods
    # =========================================================================

    def place_order(
        self,
        symbol: str,
        side: Any = None,
        order_type: Any = None,
        quantity: Decimal | None = None,
        **kwargs: Any,
    ) -> Order:
        """
        Place a trading order.

        Args:
            symbol: Trading symbol
            side: OrderSide.BUY or OrderSide.SELL
            order_type: OrderType (MARKET, LIMIT, STOP, STOP_LIMIT)
            quantity: Order quantity
            **kwargs: Additional order parameters:
                - price: Limit price
                - stop_price: Stop trigger price
                - client_id: Client order ID
                - reduce_only: Reduce-only flag
                - leverage: Position leverage

        Returns:
            Order object with execution details

        Raises:
            InsufficientFunds: If account has insufficient margin
            ValueError: If order parameters are invalid
        """
        if quantity is None:
            raise ValueError("Order quantity is required")

        # Normalize inputs
        order_side = OrderSide(side) if isinstance(side, str) else side
        ord_type = OrderType(order_type) if isinstance(order_type, str) else order_type
        price = kwargs.get("price")
        stop_price = kwargs.get("stop_price")
        client_id = kwargs.get("client_id")
        reduce_only = kwargs.get("reduce_only", False)
        leverage = kwargs.get("leverage")

        # Generate order ID
        order_id = str(uuid.uuid4())[:12]

        # Create order object
        order = Order(
            id=order_id,
            symbol=symbol,
            side=order_side,
            type=ord_type,
            quantity=Decimal(str(quantity)),
            status=OrderStatus.PENDING,
            price=Decimal(str(price)) if price else None,
            stop_price=Decimal(str(stop_price)) if stop_price else None,
            client_id=client_id,
            submitted_at=self._simulation_time,
            created_at=self._simulation_time,
        )

        # Validate sufficient funds/margin
        if not reduce_only:
            self._validate_margin(order, leverage)

        # Try immediate fill for market orders
        if ord_type == OrderType.MARKET:
            return self._execute_market_order(order, leverage)

        # For limit/stop orders, add to open orders
        self._open_orders[order_id] = order
        order.status = OrderStatus.SUBMITTED
        return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled, False if not found or already filled
        """
        if order_id not in self._open_orders:
            return False

        order = self._open_orders.pop(order_id)
        order.status = OrderStatus.CANCELLED
        order.updated_at = self._simulation_time
        self._cancelled_orders[order_id] = order
        self._order_history.append(order)

        return True

    def _validate_margin(self, order: Order, leverage: int | None) -> None:
        """Validate sufficient margin for order."""
        # Get current mark price for notional calculation
        quote = self._current_quote.get(order.symbol)
        bar = self._current_bar.get(order.symbol)

        if quote:
            price = quote.ask if order.side == OrderSide.BUY else quote.bid
        elif bar:
            price = bar.close
        elif order.price:
            price = order.price
        else:
            raise ValueError(f"No price available for {order.symbol}")

        notional = order.quantity * price
        lev = leverage or 1
        required_margin = notional / Decimal(lev)

        available = self.balances.get("USDC", Balance("USDC", Decimal("0"), Decimal("0"))).available

        if required_margin > available:
            raise InsufficientFunds(
                f"Insufficient margin: need {required_margin}, have {available}"
            )

    def _execute_market_order(self, order: Order, leverage: int | None) -> Order:
        """Execute a market order immediately."""
        bar = self._current_bar.get(order.symbol)
        quote = self._current_quote.get(order.symbol)

        if not bar and not quote:
            order.status = OrderStatus.REJECTED
            return order

        # Get bid/ask for fill model
        if quote:
            best_bid, best_ask = quote.bid, quote.ask
        elif bar:
            # Estimate spread from bar
            spread = bar.high - bar.low
            mid = bar.close
            best_bid = mid - spread / Decimal("4")
            best_ask = mid + spread / Decimal("4")
        else:
            # Should never reach here due to earlier check, but satisfy type checker
            order.status = OrderStatus.REJECTED
            return order

        # Use bar if available, otherwise create minimal bar from quote
        fill_bar = bar
        if fill_bar is None and quote:
            mid = (quote.bid + quote.ask) / 2
            fill_bar = Candle(
                ts=quote.ts,
                open=mid,
                high=quote.ask,
                low=quote.bid,
                close=mid,
                volume=Decimal("0"),
            )

        if fill_bar is None:
            order.status = OrderStatus.REJECTED
            return order

        # Simulate fill
        fill_result = self._fill_model.fill_market_order(
            order=order,
            current_bar=fill_bar,
            best_bid=best_bid,
            best_ask=best_ask,
        )

        return self._process_fill(order, fill_result, leverage)

    def _process_fill(self, order: Order, fill: FillResult, leverage: int | None) -> Order:
        """Process a fill result and update positions/balances."""
        if not fill.filled or fill.fill_price is None:
            order.status = OrderStatus.REJECTED
            return order

        # Update order
        order.status = OrderStatus.FILLED
        order.filled_quantity = fill.fill_quantity or order.quantity
        order.avg_fill_price = fill.fill_price
        order.updated_at = fill.fill_time

        # Calculate and deduct fees
        notional = order.filled_quantity * fill.fill_price
        fee = self._fee_calculator.calculate(notional, is_maker=fill.is_maker)
        self._total_fees_paid += fee

        # Track fees for completed trade calculation
        symbol = order.symbol
        if symbol in self._position_entry_fees:
            self._position_entry_fees[symbol] += fee
        else:
            self._position_entry_fees[symbol] = fee

        # Update balance for fee
        self._deduct_fee(fee)

        # Update position
        self._update_position(order, fill.fill_price, leverage)

        # Track statistics
        self._total_trades += 1
        if fill.slippage_bps:
            self._total_slippage_bps += fill.slippage_bps

        # Record to history
        self._filled_orders[order.id] = order
        self._order_history.append(order)

        return order

    def _deduct_fee(self, fee: Decimal) -> None:
        """Deduct fee from USDC balance."""
        if "USDC" in self.balances:
            balance = self.balances["USDC"]
            new_total = balance.total - fee
            new_available = balance.available - fee
            self.balances["USDC"] = Balance(
                asset="USDC",
                total=new_total,
                available=new_available,
                hold=balance.hold,
            )

    def _update_position(self, order: Order, fill_price: Decimal, leverage: int | None) -> None:
        """Update position based on filled order."""
        symbol = order.symbol
        quantity = order.filled_quantity
        is_buy = order.side == OrderSide.BUY

        existing = self.positions.get(symbol)

        if existing is None:
            # New position
            side = "long" if is_buy else "short"
            signed_quantity = quantity if is_buy else -quantity

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=signed_quantity,
                entry_price=fill_price,
                mark_price=fill_price,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side=side,
                leverage=leverage,
            )

            # Record entry time and reset entry fees for trade tracking
            if self._simulation_time is not None:
                self._position_entry_times[symbol] = self._simulation_time
            self._position_entry_fees[symbol] = Decimal("0")

            # Lock margin
            self._lock_margin(quantity * fill_price, leverage)
        else:
            # Modify existing position
            self._modify_position(existing, order, fill_price, leverage)

    def _modify_position(
        self,
        position: Position,
        order: Order,
        fill_price: Decimal,
        leverage: int | None,
    ) -> None:
        """Modify existing position with new fill."""
        symbol = order.symbol
        quantity = order.filled_quantity
        is_buy = order.side == OrderSide.BUY
        is_long = position.side == "long"

        old_quantity = abs(position.quantity)
        old_entry = position.entry_price

        # Determine if adding to or reducing position
        if (is_buy and is_long) or (not is_buy and not is_long):
            # Adding to position - calculate new average entry
            new_quantity = old_quantity + quantity
            new_entry = (old_quantity * old_entry + quantity * fill_price) / new_quantity

            signed_quantity = new_quantity if is_long else -new_quantity

            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=signed_quantity,
                entry_price=new_entry,
                mark_price=fill_price,
                unrealized_pnl=Decimal("0"),  # Will be updated on next mark
                realized_pnl=position.realized_pnl,
                side=position.side,
                leverage=leverage or position.leverage,
            )

            # Lock additional margin
            self._lock_margin(quantity * fill_price, leverage)
        else:
            # Reducing position - realize PnL
            close_quantity = min(quantity, old_quantity)

            if is_long:
                realized_pnl = (fill_price - old_entry) * close_quantity
            else:
                realized_pnl = (old_entry - fill_price) * close_quantity

            # Track win/loss
            if realized_pnl > 0:
                self._winning_trades += 1
            elif realized_pnl < 0:
                self._losing_trades += 1

            remaining_quantity = old_quantity - close_quantity

            # Release margin
            self._release_margin(close_quantity * old_entry, position.leverage)

            # Credit realized PnL to balance
            self._credit_pnl(realized_pnl)

            # Calculate fees for this trade (proportional for partial close)
            total_fees = self._position_entry_fees.get(symbol, Decimal("0"))
            close_ratio = close_quantity / old_quantity if old_quantity > 0 else Decimal("1")
            trade_fees = total_fees * close_ratio

            # Record completed trade
            entry_time = self._position_entry_times.get(symbol)
            exit_time = self._simulation_time
            if entry_time and exit_time:
                completed_trade = CompletedTrade.from_position_close(
                    trade_id=str(uuid.uuid4())[:12],
                    symbol=symbol,
                    side=position.side,
                    entry_time=entry_time,
                    entry_price=old_entry,
                    exit_time=exit_time,
                    exit_price=fill_price,
                    quantity=close_quantity,
                    fees_paid=trade_fees,
                )
                self._completed_trades.append(completed_trade)

            if remaining_quantity > Decimal("0"):
                # Partial close - update remaining fees
                self._position_entry_fees[symbol] = total_fees - trade_fees

                signed_quantity = remaining_quantity if is_long else -remaining_quantity

                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=signed_quantity,
                    entry_price=old_entry,
                    mark_price=fill_price,
                    unrealized_pnl=Decimal("0"),
                    realized_pnl=position.realized_pnl + realized_pnl,
                    side=position.side,
                    leverage=position.leverage,
                )
            else:
                # Full close - clean up tracking data
                self._position_entry_times.pop(symbol, None)
                self._position_entry_fees.pop(symbol, None)
                del self.positions[symbol]

                # Check if order flips position
                flip_quantity = quantity - old_quantity
                if flip_quantity > Decimal("0"):
                    new_side = "long" if is_buy else "short"
                    signed_flip = flip_quantity if is_buy else -flip_quantity

                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=signed_flip,
                        entry_price=fill_price,
                        mark_price=fill_price,
                        unrealized_pnl=Decimal("0"),
                        realized_pnl=Decimal("0"),
                        side=new_side,
                        leverage=leverage,
                    )
                    self._lock_margin(flip_quantity * fill_price, leverage)

                    # New position starts fresh entry tracking
                    if self._simulation_time is not None:
                        self._position_entry_times[symbol] = self._simulation_time
                    self._position_entry_fees[symbol] = Decimal("0")

    def _lock_margin(self, notional: Decimal, leverage: int | None) -> None:
        """Lock margin for a position."""
        lev = leverage or 1
        margin = notional / Decimal(lev)

        if "USDC" in self.balances:
            balance = self.balances["USDC"]
            new_available = balance.available - margin
            new_hold = balance.hold + margin
            self.balances["USDC"] = Balance(
                asset="USDC",
                total=balance.total,
                available=new_available,
                hold=new_hold,
            )

    def _release_margin(self, notional: Decimal, leverage: int | None) -> None:
        """Release margin from a closed position."""
        lev = leverage or 1
        margin = notional / Decimal(lev)

        if "USDC" in self.balances:
            balance = self.balances["USDC"]
            new_available = balance.available + margin
            new_hold = max(Decimal("0"), balance.hold - margin)
            self.balances["USDC"] = Balance(
                asset="USDC",
                total=balance.total,
                available=new_available,
                hold=new_hold,
            )

    def _credit_pnl(self, pnl: Decimal) -> None:
        """Credit realized PnL to balance."""
        if "USDC" in self.balances:
            balance = self.balances["USDC"]
            new_total = balance.total + pnl
            new_available = balance.available + pnl
            self.balances["USDC"] = Balance(
                asset="USDC",
                total=new_total,
                available=new_available,
                hold=balance.hold,
            )

    # =========================================================================
    # Extended Protocol Methods
    # =========================================================================

    def get_mark_price(self, symbol: str) -> Decimal | None:
        """Get current mark price for a symbol."""
        quote = self._current_quote.get(symbol)
        if quote:
            return quote.last

        bar = self._current_bar.get(symbol)
        if bar:
            return bar.close

        return None

    def get_market_snapshot(self, symbol: str) -> dict[str, Any]:
        """Get a snapshot of market data for a symbol."""
        quote = self._current_quote.get(symbol)
        bar = self._current_bar.get(symbol)

        snapshot: dict[str, Any] = {"symbol": symbol}

        if quote:
            snapshot.update(
                {
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "last": quote.last,
                    "spread": quote.ask - quote.bid,
                    "mid": (quote.bid + quote.ask) / Decimal("2"),
                    "timestamp": quote.ts,
                }
            )

        if bar:
            snapshot.update(
                {
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "bar_timestamp": bar.ts,
                }
            )

        return snapshot

    def get_position_pnl(self, symbol: str) -> dict[str, Decimal]:
        """Get PnL data for a position."""
        position = self.positions.get(symbol)
        if not position:
            return {
                "realized_pnl": Decimal("0"),
                "unrealized_pnl": Decimal("0"),
                "total_pnl": Decimal("0"),
            }

        return {
            "realized_pnl": position.realized_pnl,
            "unrealized_pnl": position.unrealized_pnl,
            "total_pnl": position.realized_pnl + position.unrealized_pnl,
        }

    def get_position_risk(self, symbol: str) -> dict[str, Any]:
        """Get risk metrics for a position."""
        position = self.positions.get(symbol)
        if not position:
            return {}

        mark = self.get_mark_price(symbol) or position.mark_price
        notional = abs(position.quantity) * mark
        leverage = position.leverage or 1

        # Calculate liquidation price (simplified)
        maintenance_margin_rate = Decimal("0.05")  # 5% maintenance margin
        if position.side == "long":
            liquidation_price = position.entry_price * (
                1 - Decimal("1") / Decimal(leverage) + maintenance_margin_rate
            )
        else:
            liquidation_price = position.entry_price * (
                1 + Decimal("1") / Decimal(leverage) - maintenance_margin_rate
            )

        return {
            "symbol": symbol,
            "notional": notional,
            "leverage": leverage,
            "margin_used": notional / Decimal(leverage),
            "liquidation_price": liquidation_price,
            "entry_price": position.entry_price,
            "mark_price": mark,
            "unrealized_pnl_pct": (
                position.unrealized_pnl / (notional / Decimal(leverage)) * 100
                if notional > 0
                else Decimal("0")
            ),
        }

    # =========================================================================
    # Simulation Update Methods (Called by Bar Runner)
    # =========================================================================

    def update_bar(self, symbol: str, bar: Candle) -> None:
        """
        Update current bar data for a symbol.

        Called by the bar runner when a new candle is available.
        """
        self._current_bar[symbol] = bar

        # Update candle history
        if symbol not in self._candle_history:
            self._candle_history[symbol] = []
        self._candle_history[symbol].append(bar)

        # Limit history size
        if len(self._candle_history[symbol]) > 1000:
            self._candle_history[symbol] = self._candle_history[symbol][-1000:]

        # Synthesize quote from bar
        spread = (bar.high - bar.low) / Decimal("4")
        mid = bar.close
        self._current_quote[symbol] = Quote(
            symbol=symbol,
            bid=mid - spread / Decimal("2"),
            ask=mid + spread / Decimal("2"),
            last=bar.close,
            ts=bar.ts,
        )

        # Update mark prices on positions
        self._update_position_marks(symbol, bar.close)

        # Try to fill pending limit/stop orders
        self._check_pending_orders(symbol, bar)

        # Update simulation time
        self._simulation_time = bar.ts

    def _update_position_marks(self, symbol: str, mark_price: Decimal) -> None:
        """Update mark price and unrealized PnL for a position."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        is_long = position.side == "long"
        quantity = abs(position.quantity)

        if is_long:
            unrealized = (mark_price - position.entry_price) * quantity
        else:
            unrealized = (position.entry_price - mark_price) * quantity

        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=position.quantity,
            entry_price=position.entry_price,
            mark_price=mark_price,
            unrealized_pnl=unrealized,
            realized_pnl=position.realized_pnl,
            side=position.side,
            leverage=position.leverage,
        )

    def _check_pending_orders(self, symbol: str, bar: Candle) -> None:
        """Check and fill pending limit/stop orders."""
        quote = self._current_quote.get(symbol)
        if not quote:
            return

        orders_to_process = [
            (oid, order) for oid, order in self._open_orders.items() if order.symbol == symbol
        ]

        for order_id, order in orders_to_process:
            fill_result: FillResult | None = None

            if order.type == OrderType.LIMIT:
                fill_result = self._fill_model.try_fill_limit_order(
                    order=order,
                    current_bar=bar,
                    best_bid=quote.bid,
                    best_ask=quote.ask,
                )
            elif order.type in (OrderType.STOP, OrderType.STOP_LIMIT):
                fill_result = self._fill_model.try_fill_stop_order(
                    order=order,
                    current_bar=bar,
                    best_bid=quote.bid,
                    best_ask=quote.ask,
                )

            if fill_result and fill_result.filled:
                del self._open_orders[order_id]
                # For pending orders, leverage is not tracked on Order; use None
                self._process_fill(order, fill_result, None)

    def update_equity_curve(self) -> None:
        """Record current equity to the equity curve."""
        equity = self.get_equity()
        timestamp = self._simulation_time or datetime.now()

        self._equity_curve.append((timestamp, equity))

        # Update peak and drawdown
        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = self._peak_equity - equity
        drawdown_pct = drawdown / self._peak_equity * 100 if self._peak_equity > 0 else Decimal("0")

        if drawdown > self._max_drawdown_usd:
            self._max_drawdown_usd = drawdown
            self._max_drawdown = drawdown_pct

    def process_funding(self, symbol: str, funding_rate_8h: Decimal) -> Decimal:
        """
        Process funding payment for a perpetual position.

        Args:
            symbol: Trading symbol
            funding_rate_8h: 8-hour funding rate

        Returns:
            Funding payment amount (positive = paid, negative = received)
        """
        if self._simulation_time is None:
            return Decimal("0")

        position = self.positions.get(symbol)
        if not position:
            return Decimal("0")

        mark = self.get_mark_price(symbol) or position.mark_price

        # Accrue funding
        funding = self._funding_tracker.accrue(
            symbol=symbol,
            position_size=position.quantity,
            mark_price=mark,
            funding_rate_8h=funding_rate_8h,
            current_time=self._simulation_time,
        )

        # Check for settlement
        if self._funding_tracker.should_settle(self._simulation_time, symbol):
            settled = self._funding_tracker.settle(symbol, self._simulation_time)
            # Deduct/credit funding from balance
            self._credit_pnl(-settled)  # Negative because positive funding = cost

        return funding

    # =========================================================================
    # Statistics Methods
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive trading statistics."""
        equity = self.get_equity()
        total_return = (equity - self._initial_equity) / self._initial_equity * 100

        win_rate = (
            Decimal(self._winning_trades) / Decimal(self._total_trades) * 100
            if self._total_trades > 0
            else Decimal("0")
        )

        avg_slippage = (
            self._total_slippage_bps / Decimal(self._total_trades)
            if self._total_trades > 0
            else Decimal("0")
        )

        return {
            "initial_equity": self._initial_equity,
            "final_equity": equity,
            "total_return_pct": total_return,
            "total_return_usd": equity - self._initial_equity,
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "win_rate_pct": win_rate,
            "total_fees_paid": self._total_fees_paid,
            "max_drawdown_pct": self._max_drawdown,
            "max_drawdown_usd": self._max_drawdown_usd,
            "avg_slippage_bps": avg_slippage,
            "funding_pnl": self._funding_tracker.get_total_funding_pnl(),
        }

    def get_equity_curve(self) -> list[tuple[datetime, Decimal]]:
        """Get the equity curve time series."""
        return self._equity_curve.copy()

    def get_completed_trades(self) -> list[CompletedTrade]:
        """Get all completed trades from the simulation."""
        return self._completed_trades.copy()

    def reset(self) -> None:
        """Reset broker to initial state."""
        # Preserve settings
        initial_equity = self._initial_equity
        fee_tier = self.fee_tier
        config = self.config

        # Reset account state
        self.balances = {
            "USDC": Balance(
                asset="USDC",
                total=initial_equity,
                available=initial_equity,
                hold=Decimal("0"),
            )
        }
        self.positions = {}
        self._position_history = []

        # Reset order management
        self._open_orders = {}
        self._filled_orders = {}
        self._cancelled_orders = {}
        self._order_history = []

        # Reset market state
        self._current_bar = {}
        self._current_quote = {}
        self._candle_history = {}

        # Reset components
        self._fill_model = OrderFillModel(
            slippage_bps=config.slippage_bps if config else None,
            spread_impact_pct=config.spread_impact_pct if config else Decimal("0.5"),
            limit_volume_threshold=(
                config.limit_fill_volume_threshold if config else Decimal("2.0")
            ),
        )
        self._fee_calculator = FeeCalculator(tier=fee_tier)
        self._funding_tracker = FundingPnLTracker(
            accrual_interval_hours=config.funding_accrual_hours if config else 1,
            settlement_interval_hours=config.funding_settlement_hours if config else 8,
        )

        # Reset equity tracking
        self._equity_curve = []
        self._peak_equity = initial_equity
        self._max_drawdown = Decimal("0")
        self._max_drawdown_usd = Decimal("0")

        # Reset statistics
        self._total_fees_paid = Decimal("0")
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0
        self._total_slippage_bps = Decimal("0")

        # Reset trade history tracking
        self._completed_trades = []
        self._position_entry_times = {}
        self._position_entry_fees = {}

        # Reset connection state
        self.connected = False
        self._simulation_time = None
