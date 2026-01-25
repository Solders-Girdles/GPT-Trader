"""
Hybrid Paper Broker for paper trading with real market data.

Combines real Coinbase market data with simulated order execution.
Used for paper-fills mode (real market data + simulated fills) to validate
strategies against live prices without placing exchange orders.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

from gpt_trader.core import (
    Balance,
    Candle,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Product,
    Quote,
    TimeInForce,
)
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.utilities.datetime_helpers import utc_now
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="paper_broker")


class HybridPaperBroker:
    """
    Hybrid broker: real market data + simulated order execution.

    This broker fetches live prices from Coinbase but simulates all
    order execution locally. Perfect for paper trading strategies
    against real market conditions without risking capital.

    Features:
    - Real-time prices from Coinbase public API
    - Simulated order fills with realistic slippage
    - Position tracking with PnL calculation
    - Balance management with trade accounting
    """

    def __init__(
        self,
        api_key: str | None = None,
        private_key: str | None = None,
        initial_equity: Decimal = Decimal("10000"),
        slippage_bps: int = 5,
        commission_bps: Decimal = Decimal("5"),
        client: CoinbaseClient | None = None,
    ) -> None:
        """
        Initialize the hybrid paper broker.

        Args:
            api_key: Coinbase CDP API key name (optional if client provided)
            private_key: Coinbase CDP private key (optional if client provided)
            initial_equity: Starting equity in USD
            slippage_bps: Simulated slippage in basis points
            commission_bps: Commission per trade in basis points
            client: Pre-configured CoinbaseClient (preferred)
        """
        # Real broker for market data
        if client:
            self._client = client
        elif api_key and private_key:
            auth = SimpleAuth(key_name=api_key, private_key=private_key)
            self._client = CoinbaseClient(auth=auth)
        else:
            raise ValueError("Must provide either client or (api_key, private_key)")

        # Simulation state
        self._initial_equity = initial_equity
        self._slippage_bps = slippage_bps
        self._commission_bps = commission_bps

        # Internal state
        self._positions: dict[str, Position] = {}
        self._balances: dict[str, Balance] = {
            "USD": Balance(
                asset="USD",
                total=initial_equity,
                available=initial_equity,
            ),
        }
        self._orders: dict[str, Order] = {}
        self._order_counter = 0
        self._order_id_prefix = f"{os.getpid()}_{time.time_ns()}"
        self._products_cache: dict[str, Product] = {}
        self._last_prices: dict[str, Decimal] = {}

        logger.info(
            f"HybridPaperBroker initialized with equity=${initial_equity}, "
            f"slippage={slippage_bps}bps, commission={commission_bps}bps"
        )

    # -------------------------------------------------------------------------
    # Market Data Methods (delegated to real Coinbase API)
    # -------------------------------------------------------------------------

    def get_product(self, symbol: str) -> Product | None:
        """Get product metadata from Coinbase."""
        if symbol in self._products_cache:
            return self._products_cache[symbol]

        try:
            # Use public market product endpoint
            data = self._client.get_market_product(symbol)
            product = self._parse_product(data)
            self._products_cache[symbol] = product
            return product
        except Exception as e:
            logger.warning(f"Failed to fetch product {symbol}: {e}")
            # Return synthetic product as fallback
            return self._synthetic_product(symbol)

    def get_quote(self, symbol: str) -> Quote | None:
        """Get current quote from Coinbase."""
        try:
            ticker = self._client.get_market_product_ticker(symbol)

            # Parse best bid/ask from ticker
            bid = Decimal(ticker.get("best_bid", "0"))
            ask = Decimal(ticker.get("best_ask", "0"))

            # Get last trade price from trades array
            trades = ticker.get("trades", [])
            if trades and len(trades) > 0:
                price = Decimal(trades[0].get("price", "0"))
            else:
                # Fallback to mid price if no trades
                price = (bid + ask) / 2 if bid > 0 and ask > 0 else Decimal("0")

            self._last_prices[symbol] = price

            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=price,
                ts=datetime.now(),
            )
        except Exception as e:
            logger.warning(f"Failed to fetch quote for {symbol}: {e}")
            return None

    def get_ticker(self, product_id: str) -> dict[str, Any]:
        """Get ticker data from Coinbase."""
        try:
            get_ticker = getattr(self._client, "get_ticker", None)
            if callable(get_ticker):
                response = get_ticker(product_id)
            else:
                response = self._client.get_market_product_ticker(product_id)
        except Exception as e:
            logger.warning(f"Failed to fetch ticker for {product_id}: {e}")
            return {}

        price = response.get("price")
        if not price:
            trades = response.get("trades", [])
            if trades:
                price = trades[0].get("price")
        bid = response.get("bid") or response.get("best_bid")
        ask = response.get("ask") or response.get("best_ask")

        normalized = {
            "product_id": product_id,
            "price": price or "0",
            "bid": bid or "0",
            "ask": ask or "0",
            **response,
        }
        try:
            price_dec = Decimal(str(normalized.get("price", "0")))
            if price_dec > 0:
                self._last_prices[product_id] = price_dec
        except Exception:
            pass
        return normalized

    def get_candles(
        self, symbol: str, granularity: str = "ONE_HOUR", limit: int = 200
    ) -> list[Candle]:
        """Get historical candles from Coinbase."""
        try:
            data = self._client.get_market_product_candles(symbol, granularity, limit)
            candles_raw = data.get("candles", [])
            return [self._parse_candle(c) for c in candles_raw]
        except Exception as e:
            logger.warning(f"Failed to fetch candles for {symbol}: {e}")
            return []

    def list_products(self, product_type: str | None = None) -> list[Product]:
        """List available products from Coinbase."""
        try:
            data = self._client.get_market_products()
            products_raw = data.get("products", [])
            products = [self._parse_product(p) for p in products_raw if p]

            if product_type:
                target = product_type.upper()
                products = [p for p in products if p.market_type.value.upper() == target]

            return products
        except Exception as e:
            logger.warning(f"Failed to list products: {e}")
            return []

    # -------------------------------------------------------------------------
    # Position & Balance Methods (simulated state)
    # -------------------------------------------------------------------------

    def list_positions(self) -> list[Position]:
        """Return simulated positions."""
        # Keep marks fresh for downstream strategy code without forcing extra API calls.
        for pos in self._positions.values():
            last_price = self._last_prices.get(pos.symbol)
            if last_price is not None:
                pos.mark_price = last_price
                # EquityCalculator already values spot holdings via balances, so avoid
                # double-counting PnL by keeping spot positions' unrealized_pnl at 0.
                pos.unrealized_pnl = Decimal("0")
        return list(self._positions.values())

    def get_positions(self) -> list[Position]:
        """Alias for list_positions."""
        return self.list_positions()

    def list_balances(self) -> list[Balance]:
        """Return simulated balances."""
        return list(self._balances.values())

    def get_balances(self) -> list[Balance]:
        """Alias for list_balances."""
        return self.list_balances()

    def get_equity(self) -> Decimal:
        """Calculate total equity from simulated balances and cached prices."""
        total = Decimal("0")
        for balance in self._balances.values():
            asset = str(balance.asset or "").upper()
            amount = balance.total
            if amount == 0:
                continue
            if asset in {"USD", "USDC"}:
                total += amount
                continue

            product_id = f"{asset}-USD"
            price = self._last_prices.get(product_id)
            if price is None:
                ticker = self.get_ticker(product_id)
                try:
                    price = Decimal(str(ticker.get("price", "0")))
                except Exception:
                    price = Decimal("0")
            total += amount * price
        return total

    # -------------------------------------------------------------------------
    # Order Execution Methods (simulated)
    # -------------------------------------------------------------------------

    def place_order(
        self,
        symbol_or_payload: str | dict[str, Any] | None = None,
        side: str | OrderSide | None = None,
        order_type: str | OrderType = "market",
        quantity: Decimal | None = None,
        limit_price: Decimal | None = None,
        **kwargs: Any,
    ) -> Order:
        """
        Place a simulated order with realistic fill simulation.

        Orders are filled immediately at current market price with slippage.
        """
        self._order_counter += 1
        order_id = f"PAPER_{self._order_id_prefix}_{self._order_counter:06d}"

        # Handle 'symbol' keyword arg (from BrokerExecutor)
        if symbol_or_payload is None:
            symbol_or_payload = kwargs.pop("symbol", "")

        # Handle 'price' keyword arg (alias for limit_price, from BrokerExecutor)
        if limit_price is None and "price" in kwargs:
            limit_price = kwargs.pop("price")

        # Handle dict payload from strategy engine
        if isinstance(symbol_or_payload, dict):
            payload = symbol_or_payload
            symbol = payload.get("product_id") or payload.get("symbol") or ""
            side = payload.get("side", side) or "BUY"
            order_config = payload.get("order_configuration", {})
            if "market_market_ioc" in order_config:
                quote_size = order_config["market_market_ioc"].get("quote_size")
                base_size = order_config["market_market_ioc"].get("base_size")
                if base_size:
                    quantity = Decimal(base_size)
                elif quote_size:
                    price = self._get_current_price(symbol)
                    quantity = Decimal(quote_size) / price if price else Decimal("0.001")
        else:
            symbol = str(symbol_or_payload or "")

        # Convert strings to enums
        if isinstance(side, str):
            side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        else:
            side_enum = side or OrderSide.BUY

        if isinstance(order_type, str):
            type_enum = OrderType.MARKET if order_type.lower() == "market" else OrderType.LIMIT
        else:
            type_enum = order_type

        # Get current price for fill
        current_price = self._get_current_price(symbol)
        if current_price is None:
            logger.error(f"Cannot get price for {symbol}, order rejected")
            return self._rejected_order(order_id, symbol, side_enum, quantity or Decimal("0"))

        # Apply slippage
        slippage_mult = Decimal(1) + Decimal(self._slippage_bps) / Decimal("10000")
        if side_enum == OrderSide.BUY:
            fill_price = current_price * slippage_mult
        else:
            fill_price = current_price / slippage_mult

        # Use limit price if better
        if limit_price and type_enum == OrderType.LIMIT:
            if side_enum == OrderSide.BUY and limit_price < fill_price:
                fill_price = limit_price
            elif side_enum == OrderSide.SELL and limit_price > fill_price:
                fill_price = limit_price

        fill_quantity = quantity or Decimal("0.001")
        if fill_quantity <= 0:
            return self._rejected_order(order_id, symbol, side_enum, Decimal("0"))
        now = utc_now()

        # Calculate commission
        notional = fill_price * fill_quantity
        commission = notional * self._commission_bps / Decimal("10000")

        reduce_only = bool(kwargs.get("reduce_only", False))
        base_asset, quote_asset = self._parse_symbol(symbol)

        if side_enum == OrderSide.BUY:
            required = notional + commission
            if self._get_available(quote_asset) < required:
                logger.info(
                    "[PAPER] Insufficient %s for BUY (required=%s, available=%s)",
                    quote_asset,
                    required,
                    self._get_available(quote_asset),
                )
                return self._rejected_order(order_id, symbol, side_enum, fill_quantity)
        else:
            available_base = self._get_available(base_asset)
            if available_base <= 0:
                return self._rejected_order(order_id, symbol, side_enum, fill_quantity)
            if fill_quantity > available_base:
                if reduce_only:
                    fill_quantity = available_base
                    notional = fill_price * fill_quantity
                    commission = notional * self._commission_bps / Decimal("10000")
                else:
                    return self._rejected_order(order_id, symbol, side_enum, fill_quantity)

        # Update simulated balances + positions after acceptance checks.
        self._update_balances_from_fill(
            symbol=symbol,
            side=side_enum,
            quantity=fill_quantity,
            fill_price=fill_price,
            commission=commission,
        )
        self._update_position_from_fill(
            symbol=symbol,
            side=side_enum,
            quantity=fill_quantity,
            fill_price=fill_price,
        )

        order = Order(
            id=order_id,
            client_id=kwargs.get("client_id", order_id),
            symbol=symbol,
            side=side_enum,
            type=type_enum,
            quantity=fill_quantity,
            price=fill_price,
            stop_price=kwargs.get("stop_price"),
            tif=kwargs.get("tif", TimeInForce.GTC),
            status=OrderStatus.FILLED,
            filled_quantity=fill_quantity,
            avg_fill_price=fill_price,
            submitted_at=now,
            updated_at=now,
        )

        self._orders[order_id] = order

        logger.info(
            f"[PAPER] {side_enum.value} {fill_quantity} {symbol} @ ${fill_price:.2f} "
            f"(commission: ${commission:.2f})"
        )

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a simulated order (always succeeds for paper trading)."""
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            logger.info(f"[PAPER] Cancelled order {order_id}")
            return True
        return False

    def get_order(self, order_id: str) -> Order | None:
        """Get a simulated order by ID."""
        return self._orders.get(order_id)

    # -------------------------------------------------------------------------
    # Market Data Protocol Methods
    # -------------------------------------------------------------------------

    def start_market_data(self, symbols: list[str]) -> None:
        """No-op for paper broker (uses REST polling)."""
        logger.info(f"[PAPER] Market data tracking started for {symbols}")
        # Pre-fetch initial prices
        for symbol in symbols:
            self.get_quote(symbol)

    def stop_market_data(self) -> None:
        """No-op for paper broker."""
        pass

    def is_connected(self) -> bool:
        """Always connected for paper trading."""
        return True

    def is_stale(self, symbol: str, threshold_seconds: int = 30) -> bool:
        """Paper broker is never stale (fetches on demand)."""
        return False

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_current_price(self, symbol: str) -> Decimal | None:
        """Get current price, either from cache or fresh fetch."""
        if symbol in self._last_prices:
            return self._last_prices[symbol]

        quote = self.get_quote(symbol)
        return quote.last if quote else None

    def _parse_symbol(self, symbol: str) -> tuple[str, str]:
        base, _, quote = symbol.partition("-")
        return base.upper(), (quote or "USD").upper()

    def _ensure_balance(self, asset: str) -> Balance:
        asset_upper = asset.upper()
        existing = self._balances.get(asset_upper)
        if existing is None:
            existing = Balance(asset=asset_upper, total=Decimal("0"), available=Decimal("0"))
            self._balances[asset_upper] = existing
        return existing

    def _get_available(self, asset: str) -> Decimal:
        balance = self._balances.get(asset.upper())
        return balance.available if balance is not None else Decimal("0")

    def _adjust_balance(self, asset: str, delta: Decimal) -> None:
        bal = self._ensure_balance(asset)
        asset_upper = asset.upper()
        self._balances[asset_upper] = Balance(
            asset=asset_upper,
            total=bal.total + delta,
            available=bal.available + delta,
            hold=bal.hold,
        )

    def _update_balances_from_fill(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        fill_price: Decimal,
        commission: Decimal,
    ) -> None:
        base_asset, quote_asset = self._parse_symbol(symbol)
        notional = fill_price * quantity
        if side == OrderSide.BUY:
            self._adjust_balance(quote_asset, -(notional + commission))
            self._adjust_balance(base_asset, quantity)
        else:
            self._adjust_balance(base_asset, -quantity)
            self._adjust_balance(quote_asset, notional - commission)

    def _update_position_from_fill(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        fill_price: Decimal,
    ) -> None:
        if quantity <= 0:
            return

        position = self._positions.get(symbol)
        if position is None:
            if side != OrderSide.BUY:
                return
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=fill_price,
                mark_price=fill_price,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                side="long",
                leverage=1,
            )
            return

        if side == OrderSide.BUY:
            new_quantity = position.quantity + quantity
            if new_quantity <= 0:
                return
            weighted_entry = (
                position.entry_price * position.quantity + fill_price * quantity
            ) / new_quantity
            position.quantity = new_quantity
            position.entry_price = weighted_entry
            position.mark_price = fill_price
            position.unrealized_pnl = Decimal("0")
            position.side = "long"
            return

        # SELL reduces / closes a long spot position.
        closing_quantity = min(position.quantity, quantity)
        realized_delta = (fill_price - position.entry_price) * closing_quantity
        remaining = position.quantity - closing_quantity
        if remaining <= 0:
            self._positions.pop(symbol, None)
            return
        position.quantity = remaining
        position.mark_price = fill_price
        position.realized_pnl += realized_delta
        position.unrealized_pnl = Decimal("0")

    def _parse_product(self, data: dict[str, Any]) -> Product:
        """Parse product from API response."""
        symbol = data.get("product_id", "UNKNOWN")
        product_type = data.get("product_type", "SPOT").upper()

        return Product(
            symbol=symbol,
            base_asset=data.get("base_currency_id", symbol.split("-")[0]),
            quote_asset=data.get("quote_currency_id", "USD"),
            market_type=MarketType.PERPETUAL if "PERP" in product_type else MarketType.SPOT,
            min_size=Decimal(data.get("base_min_size", "0.001")),
            step_size=Decimal(data.get("base_increment", "0.001")),
            min_notional=Decimal(data.get("min_market_funds", "10")),
            price_increment=Decimal(data.get("quote_increment", "0.01")),
            leverage_max=1,
        )

    def _parse_candle(self, data: dict[str, Any]) -> Candle:
        """Parse candle from API response."""
        return Candle(
            ts=datetime.fromtimestamp(int(data.get("start", 0))),
            open=Decimal(data.get("open", "0")),
            high=Decimal(data.get("high", "0")),
            low=Decimal(data.get("low", "0")),
            close=Decimal(data.get("close", "0")),
            volume=Decimal(data.get("volume", "0")),
        )

    def _synthetic_product(self, symbol: str) -> Product:
        """Create a synthetic product definition."""
        base, _, quote = symbol.partition("-")
        quote = quote or "USD"

        return Product(
            symbol=symbol,
            base_asset=base,
            quote_asset=quote,
            market_type=MarketType.SPOT,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=1,
        )

    def _rejected_order(
        self, order_id: str, symbol: str, side: OrderSide, quantity: Decimal
    ) -> Order:
        """Create a rejected order response."""
        now = utc_now()
        return Order(
            id=order_id,
            client_id=order_id,
            symbol=symbol,
            side=side,
            type=OrderType.MARKET,
            quantity=quantity,
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.REJECTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=now,
            updated_at=now,
        )

    # -------------------------------------------------------------------------
    # Status & Debugging
    # -------------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get current paper trading status."""
        return {
            "mode": "paper",
            "initial_equity": float(self._initial_equity),
            "current_equity": float(self.get_equity()),
            "cash_balance": float(
                self._balances.get(
                    "USD", Balance(asset="USD", total=Decimal("0"), available=Decimal("0"))
                ).total
            ),
            "positions": len(self._positions),
            "orders_executed": self._order_counter,
            "last_prices": {k: float(v) for k, v in self._last_prices.items()},
        }


__all__ = ["HybridPaperBroker"]
