"""
Minimal test adapter for Week 1 validation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal

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
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.models import APIConfig, to_product
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog

logger = logging.getLogger(__name__)


class MinimalCoinbaseBrokerage:
    """Minimal adapter for Week 1 validation testing."""

    def __init__(self, config: APIConfig):
        self.config = config
        self.endpoints = CoinbaseEndpoints(config)
        self.product_catalog = ProductCatalog(ttl_seconds=900)

        # Mock WebSocket data for testing
        self._ws_client = None
        self._market_data: dict[str, dict] = {}
        self._rolling_windows: dict[str, dict] = {}

        logger.info(f"MinimalCoinbaseBrokerage initialized - sandbox: {config.sandbox}")

    def list_products(self, market: MarketType | None = None) -> list[Product]:
        """Mock product listing for validation."""
        # Return mock perpetuals for testing
        mock_products_data = [
            {
                "product_id": "BTC-PERP",
                "display_name": "Bitcoin Perpetual",
                "status": "online",
                "base_currency_id": "BTC",
                "quote_currency_id": "USD",
                "base_min_size": "0.0001",
                "base_max_size": "10000",
                "quote_min_size": "1",
                "quote_max_size": "1000000",
                "base_increment": "0.0001",
                "quote_increment": "0.01",
                "product_type": "FUTURE",
                "contract_type": "perpetual",
                "future_product_details": {
                    "venue": "CFM",
                    "contract_code": "BTC",
                    "contract_expiry": "PERPETUAL",
                    "contract_size": "1",
                    "contract_root_unit": "BTC",
                    "group_description": "BTC Perpetual",
                    "contract_expiry_timezone": "UTC",
                    "group_short_description": "BTC-PERP",
                    "funding_rate": "0.0001",
                    "funding_time": "2024-01-01T08:00:00Z",
                },
            },
            {
                "product_id": "ETH-PERP",
                "display_name": "Ethereum Perpetual",
                "status": "online",
                "base_currency_id": "ETH",
                "quote_currency_id": "USD",
                "base_min_size": "0.001",
                "base_max_size": "10000",
                "quote_min_size": "1",
                "quote_max_size": "1000000",
                "base_increment": "0.001",
                "quote_increment": "0.01",
                "product_type": "FUTURE",
                "contract_type": "perpetual",
                "future_product_details": {
                    "venue": "CFM",
                    "contract_code": "ETH",
                    "contract_expiry": "PERPETUAL",
                    "contract_size": "1",
                    "contract_root_unit": "ETH",
                    "group_description": "ETH Perpetual",
                    "contract_expiry_timezone": "UTC",
                    "group_short_description": "ETH-PERP",
                    "funding_rate": "0.0002",
                    "funding_time": "2024-01-01T08:00:00Z",
                },
            },
        ]

        products = []
        for item in mock_products_data:
            product = to_product(item)
            if market is None or product.market_type == market:
                products.append(product)

        logger.info(f"Mock: Listed {len(products)} products")
        return products

    def get_product(self, symbol: str) -> Product | None:
        """Mock get single product."""
        products = self.list_products()
        for p in products:
            if p.symbol == symbol:
                return p
        return None

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "market",
        quantity: Decimal | None = None,
        limit_price: Decimal | None = None,
        **kwargs,
    ) -> Order | None:
        """Mock order placement for validation."""
        # Convert strings to enums
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        type_enum = OrderType.MARKET if order_type.lower() == "market" else OrderType.LIMIT

        # Mock successful order
        mock_order = Order(
            id=f"mock_{symbol}_{side}_{order_type}",
            client_id=f"mock_client_{symbol}",
            symbol=symbol,
            side=side_enum,
            type=type_enum,
            quantity=quantity or Decimal("0.01"),
            price=limit_price or Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        logger.info(f"Mock: Placed {order_type} {side} order for {symbol}")
        return mock_order

    def cancel_order(self, order_id: str) -> bool:
        """Mock order cancellation."""
        logger.info(f"Mock: Cancelled order {order_id}")
        return True

    def get_quote(self, symbol: str) -> Quote:
        """Mock quote for validation."""
        from ..core.interfaces import Quote

        # Mock BTC quote
        if symbol.startswith("BTC"):
            price = Decimal("50000")
        elif symbol.startswith("ETH"):
            price = Decimal("3000")
        else:
            price = Decimal("100")

        spread = price * Decimal("0.0001")  # 1 bps spread

        mock_quote = Quote(
            symbol=symbol,
            bid=price - spread / 2,
            ask=price + spread / 2,
            last=price,
            ts=datetime.now(),
        )

        return mock_quote

    def get_candles(self, symbol: str, granularity: str, limit: int = 200) -> list[Candle]:
        """Mock candles."""
        return []

    def get_balances(self) -> list[Balance]:
        """Mock balances."""
        return []

    def get_positions(self) -> list[Position]:
        """Mock positions."""
        return []

    def is_connected(self) -> bool:
        """Always connected for testing."""
        return True

    def start_market_data(self, symbols: list[str]) -> None:
        """Mock WebSocket market data start."""
        self._ws_client = "mock_ws_client"  # Mock connection

        for symbol in symbols:
            # Initialize market data
            self._market_data[symbol] = {
                "mid": Decimal("50000") if symbol.startswith("BTC") else Decimal("3000"),
                "spread_bps": 10.0,  # 10 bps
                "depth_l1": Decimal("100000"),  # USD notional
                "depth_l10": Decimal("1000000"),  # USD notional
                "last_update": "2024-01-01T00:00:00Z",
            }

            # Initialize rolling windows
            self._rolling_windows[symbol] = {"vol_1m": 0.0, "vol_5m": 0.0}

        logger.info(f"Mock: Started market data for {symbols}")

    def is_stale(self, symbol: str, threshold_seconds: int = 10) -> bool:
        """Mock staleness check."""
        # Return False (not stale) if we have market data
        return symbol not in self._market_data

    def get_market_snapshot(self, symbol: str) -> dict:
        """Mock market snapshot."""
        return {
            "vol_1m": 100.5,  # Mock volume
            "vol_5m": 500.0,
            "spread_bps": 10.0,
            "depth_l1": Decimal("100000"),  # USD notional
            "depth_l10": Decimal("1000000"),  # USD notional
        }
