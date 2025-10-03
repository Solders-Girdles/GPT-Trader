"""Coinbase API model factories for testing.

Provides factory functions to generate realistic Coinbase API payloads that can be
mapped to internal Product, Quote, Order, Position, and Candle types via the
to_* mapper functions in bot_v2.features.brokerages.coinbase.models.

These factories reduce test boilerplate and fixture drift while enabling systematic
testing of edge cases (invalid enums, min-notional, GTD fallback, etc.).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any


class CoinbaseProductFactory:
    """Factory for creating Coinbase product API payloads."""

    @staticmethod
    def create_spot_product(
        symbol: str = "BTC-USD",
        base_currency: str = "BTC",
        quote_currency: str = "USD",
        base_min_size: str = "0.00001",
        base_increment: str = "0.00000001",
        quote_increment: str = "0.01",
        min_notional: str | None = None,
        **overrides,
    ) -> dict[str, Any]:
        """Create a spot product payload."""
        product = {
            "product_id": symbol,
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "base_min_size": base_min_size,
            "base_increment": base_increment,
            "quote_increment": quote_increment,
        }
        if min_notional:
            product["min_notional"] = min_notional
        product.update(overrides)
        return product

    @staticmethod
    def create_perps_product(
        symbol: str = "BTC-USD-PERP",
        base_currency: str = "BTC",
        quote_currency: str = "USD",
        contract_type: str = "perpetual",
        contract_size: str = "1",
        max_leverage: int = 10,
        funding_rate: str = "0.0001",
        next_funding_time: str | None = None,
        min_size: str = "0.001",
        step_size: str = "0.001",
        price_increment: str = "0.5",
        **overrides,
    ) -> dict[str, Any]:
        """Create a perpetual futures product payload."""
        if next_funding_time is None:
            next_funding_time = (datetime.utcnow() + timedelta(hours=8)).isoformat()

        product = {
            "product_id": symbol,
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "contract_type": contract_type,
            "contract_size": contract_size,
            "max_leverage": max_leverage,
            "funding_rate": funding_rate,
            "next_funding_time": next_funding_time,
            "min_size": min_size,
            "step_size": step_size,
            "price_increment": price_increment,
        }
        product.update(overrides)
        return product

    @staticmethod
    def create_futures_product(
        symbol: str = "BTC-USD-FUT-250331",
        base_currency: str = "BTC",
        quote_currency: str = "USD",
        contract_type: str = "future",
        expiry: str | None = None,
        contract_size: str = "1",
        max_leverage: int = 5,
        **overrides,
    ) -> dict[str, Any]:
        """Create a dated futures product payload."""
        if expiry is None:
            expiry = (datetime.utcnow() + timedelta(days=90)).isoformat()

        product = {
            "product_id": symbol,
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "contract_type": contract_type,
            "expiry": expiry,
            "contract_size": contract_size,
            "max_leverage": max_leverage,
            "min_size": "0.001",
            "step_size": "0.001",
            "price_increment": "0.5",
        }
        product.update(overrides)
        return product


class CoinbaseQuoteFactory:
    """Factory for creating Coinbase quote/ticker API payloads."""

    @staticmethod
    def create_quote(
        symbol: str = "BTC-USD",
        bid: str = "49950.00",
        ask: str = "50050.00",
        price: str = "50000.00",
        time: str | None = None,
        **overrides,
    ) -> dict[str, Any]:
        """Create a quote/ticker payload."""
        if time is None:
            time = datetime.utcnow().isoformat()

        quote = {
            "product_id": symbol,
            "best_bid": bid,
            "best_ask": ask,
            "price": price,
            "time": time,
        }
        quote.update(overrides)
        return quote

    @staticmethod
    def create_quote_from_trades(
        symbol: str = "BTC-USD",
        last_trade_price: str = "50000.00",
        last_trade_time: str | None = None,
        **overrides,
    ) -> dict[str, Any]:
        """Create a quote payload that derives last price from trades array."""
        if last_trade_time is None:
            last_trade_time = datetime.utcnow().isoformat()

        quote = {
            "product_id": symbol,
            "trades": [{"price": last_trade_price, "time": last_trade_time}],
        }
        quote.update(overrides)
        return quote


class CoinbaseCandleFactory:
    """Factory for creating Coinbase candle/OHLCV API payloads."""

    @staticmethod
    def create_candle(
        time: str | None = None,
        open: str = "50000.00",
        high: str = "50500.00",
        low: str = "49500.00",
        close: str = "50200.00",
        volume: str = "1000.5",
        **overrides,
    ) -> dict[str, Any]:
        """Create a candle payload."""
        if time is None:
            time = datetime.utcnow().isoformat()

        candle = {
            "time": time,
            "open": open,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        candle.update(overrides)
        return candle

    @staticmethod
    def create_candle_series(
        count: int = 10,
        interval_minutes: int = 5,
        initial_price: float = 50000.0,
        volatility: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Create a series of candles with realistic price movement."""
        import random

        candles = []
        current_time = datetime.utcnow()
        current_price = initial_price

        for i in range(count):
            # Generate random OHLC
            open_price = current_price
            price_change = current_price * volatility * random.uniform(-1, 1)
            close_price = current_price + price_change

            high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility / 2))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility / 2))
            volume = random.uniform(100, 1000)

            candle = CoinbaseCandleFactory.create_candle(
                time=(
                    current_time - timedelta(minutes=interval_minutes * (count - i - 1))
                ).isoformat(),
                open=f"{open_price:.2f}",
                high=f"{high_price:.2f}",
                low=f"{low_price:.2f}",
                close=f"{close_price:.2f}",
                volume=f"{volume:.2f}",
            )
            candles.append(candle)
            current_price = close_price

        return candles


class CoinbaseOrderFactory:
    """Factory for creating Coinbase order API payloads."""

    @staticmethod
    def create_order(
        order_id: str = "test-order-123",
        product_id: str = "BTC-USD",
        side: str = "buy",
        order_type: str = "limit",
        price: str | None = "50000.00",
        size: str = "0.01",
        status: str = "open",
        time_in_force: str = "gtc",
        client_order_id: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        filled_size: str = "0",
        average_filled_price: str | None = None,
        **overrides,
    ) -> dict[str, Any]:
        """Create a standard order payload."""
        if created_at is None:
            created_at = datetime.utcnow().isoformat()
        if updated_at is None:
            updated_at = created_at

        order = {
            "order_id": order_id,
            "product_id": product_id,
            "side": side,
            "type": order_type,
            "size": size,
            "status": status,
            "time_in_force": time_in_force,
            "created_at": created_at,
            "updated_at": updated_at,
            "filled_size": filled_size,
        }

        if price is not None:
            order["price"] = price
        if client_order_id is not None:
            order["client_order_id"] = client_order_id
        if average_filled_price is not None:
            order["average_filled_price"] = average_filled_price

        order.update(overrides)
        return order

    @staticmethod
    def create_market_order(
        order_id: str = "market-order-123",
        product_id: str = "BTC-USD",
        side: str = "buy",
        size: str = "0.01",
        status: str = "filled",
        filled_size: str | None = None,
        average_filled_price: str = "50000.00",
        **overrides,
    ) -> dict[str, Any]:
        """Create a market order payload."""
        if filled_size is None:
            filled_size = size

        return CoinbaseOrderFactory.create_order(
            order_id=order_id,
            product_id=product_id,
            side=side,
            order_type="market",
            price=None,
            size=size,
            status=status,
            time_in_force="ioc",
            filled_size=filled_size,
            average_filled_price=average_filled_price,
            **overrides,
        )

    @staticmethod
    def create_stop_order(
        order_id: str = "stop-order-123",
        product_id: str = "BTC-USD",
        side: str = "sell",
        stop_price: str = "49000.00",
        size: str = "0.01",
        status: str = "pending",
        **overrides,
    ) -> dict[str, Any]:
        """Create a stop order payload."""
        return CoinbaseOrderFactory.create_order(
            order_id=order_id,
            product_id=product_id,
            side=side,
            order_type="stop_market",
            price=None,
            size=size,
            status=status,
            stop_price=stop_price,
            **overrides,
        )

    @staticmethod
    def create_perps_order(
        order_id: str = "perps-order-123",
        product_id: str = "BTC-USD-PERP",
        side: str = "buy",
        contracts: str = "10",
        leverage: str = "5",
        **overrides,
    ) -> dict[str, Any]:
        """Create a perpetual futures order payload."""
        order = CoinbaseOrderFactory.create_order(
            order_id=order_id,
            product_id=product_id,
            side=side,
            size=contracts,  # Use size field for contracts
            **overrides,
        )
        order["contracts"] = contracts
        order["leverage"] = leverage
        return order


class CoinbasePositionFactory:
    """Factory for creating Coinbase position API payloads."""

    @staticmethod
    def create_position(
        product_id: str = "BTC-USD-PERP",
        side: str = "long",
        size: str = "10",
        entry_price: str = "50000.00",
        mark_price: str = "50500.00",
        unrealized_pnl: str | None = None,
        realized_pnl: str = "0",
        leverage: str = "5",
        **overrides,
    ) -> dict[str, Any]:
        """Create a position payload."""
        # Calculate unrealized PnL if not provided
        if unrealized_pnl is None:
            qty = Decimal(size)
            entry = Decimal(entry_price)
            mark = Decimal(mark_price)
            multiplier = Decimal("1") if side == "long" else Decimal("-1")
            unrealized_pnl = str(qty * (mark - entry) * multiplier)

        position = {
            "product_id": product_id,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "mark_price": mark_price,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "leverage": leverage,
        }
        position.update(overrides)
        return position

    @staticmethod
    def create_spot_position(
        product_id: str = "BTC-USD",
        size: str = "1.5",
        entry_price: str = "50000.00",
        current_price: str = "50500.00",
        **overrides,
    ) -> dict[str, Any]:
        """Create a spot position payload (for exchanges that track spot holdings as positions)."""
        # For spot, always long
        qty = Decimal(size)
        entry = Decimal(entry_price)
        current = Decimal(current_price)
        unrealized_pnl = str(qty * (current - entry))

        return CoinbasePositionFactory.create_position(
            product_id=product_id,
            side="long",
            size=size,
            entry_price=entry_price,
            mark_price=current_price,
            unrealized_pnl=unrealized_pnl,
            leverage="1",
            **overrides,
        )


class CoinbaseBalanceFactory:
    """Factory for creating Coinbase balance/account API payloads."""

    @staticmethod
    def create_balance(
        currency: str = "USD",
        available: str = "10000.00",
        hold: str = "0.00",
        **overrides,
    ) -> dict[str, Any]:
        """Create a balance payload."""
        total = str(Decimal(available) + Decimal(hold))

        balance = {
            "currency": currency,
            "available": available,
            "hold": hold,
            "total": total,
        }
        balance.update(overrides)
        return balance

    @staticmethod
    def create_account(
        account_id: str = "test-account-123",
        currency: str = "USD",
        balance: str = "10000.00",
        available: str | None = None,
        hold: str = "0.00",
        **overrides,
    ) -> dict[str, Any]:
        """Create an account payload."""
        if available is None:
            available = str(Decimal(balance) - Decimal(hold))

        account = {
            "id": account_id,
            "currency": currency,
            "balance": balance,
            "available": available,
            "hold": hold,
        }
        account.update(overrides)
        return account


# Edge case factories for testing validation and error handling


class CoinbaseEdgeCaseFactory:
    """Factory for creating edge case and invalid payloads."""

    @staticmethod
    def create_invalid_product(invalid_field: str = "contract_type") -> dict[str, Any]:
        """Create a product with an invalid enum value."""
        product = CoinbaseProductFactory.create_perps_product()
        if invalid_field == "contract_type":
            product["contract_type"] = "invalid_type"
        return product

    @staticmethod
    def create_min_notional_violation_order(
        product_id: str = "BTC-USD",
        price: str = "50000.00",
        size: str = "0.00001",  # Very small size
    ) -> dict[str, Any]:
        """Create an order that violates min notional requirements."""
        return CoinbaseOrderFactory.create_order(
            product_id=product_id,
            price=price,
            size=size,
        )

    @staticmethod
    def create_gtd_fallback_order(
        order_id: str = "gtd-order-123",
        product_id: str = "BTC-USD",
        time_in_force: str = "gtd",
        **overrides,
    ) -> dict[str, Any]:
        """Create an order with GTD time-in-force (should fallback to GTC)."""
        return CoinbaseOrderFactory.create_order(
            order_id=order_id,
            product_id=product_id,
            time_in_force=time_in_force,
            **overrides,
        )

    @staticmethod
    def create_missing_fields_order() -> dict[str, Any]:
        """Create an order with missing required fields to test fallbacks."""
        return {
            "order_id": "incomplete-order",
            # Missing most fields
        }

    @staticmethod
    def create_zero_price_quote(symbol: str = "BTC-USD") -> dict[str, Any]:
        """Create a quote with zero/missing price (should fallback to trades)."""
        return {
            "product_id": symbol,
            "price": "0",
            "trades": [{"price": "50000.00", "time": datetime.utcnow().isoformat()}],
        }
