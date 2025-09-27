"""
Minimal test adapter for Week 1 validation.
"""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional

from .models import APIConfig
from .endpoints import CoinbaseEndpoints
from .utils import ProductCatalog
from ..core.interfaces import (
    IBrokerage, MarketType, Product, Order, Quote, Candle, Position, Balance
)

import logging
logger = logging.getLogger(__name__)


class MinimalCoinbaseBrokerage(IBrokerage):
    """Minimal adapter for Week 1 validation testing."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.endpoints = CoinbaseEndpoints(
            mode=config.api_mode,
            sandbox=config.sandbox,
            enable_derivatives=config.enable_derivatives
        )
        self.product_catalog = ProductCatalog(ttl_seconds=900)
        
        # Mock WebSocket data for testing
        self._ws_client = None
        self._market_data: Dict[str, Dict] = {}
        self._rolling_windows: Dict[str, Dict] = {}
        
        logger.info(f"MinimalCoinbaseBrokerage initialized - sandbox: {config.sandbox}")
    
    def list_products(self, market: Optional[MarketType] = None) -> List[Product]:
        """Mock product listing for validation."""
        # Return mock perpetuals for testing
        from .models import to_product
        
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
                    "funding_time": "2024-01-01T08:00:00Z"
                }
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
                    "funding_time": "2024-01-01T08:00:00Z"
                }
            }
        ]
        
        products = []
        for item in mock_products_data:
            product = to_product(item)
            if market is None or product.market_type == market:
                products.append(product)
        
        logger.info(f"Mock: Listed {len(products)} products")
        return products
    
    def get_product(self, symbol: str) -> Optional[Product]:
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
        quantity: Optional[Decimal] = None,
        limit_price: Optional[Decimal] = None,
        **kwargs
    ) -> Optional[Order]:
        """Mock order placement for validation."""
        from ..core.interfaces import Order, OrderSide, OrderType, OrderStatus, TimeInForce
        
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
            qty=quantity or Decimal("0.01"),
            price=limit_price or Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.SUBMITTED,
            filled_qty=Decimal("0"),
            avg_fill_price=None,
            submitted_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z"
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
            bid=price - spread/2,
            ask=price + spread/2,
            last=price,
            ts=datetime.now()
        )
        
        return mock_quote
    
    def get_candles(self, symbol: str, granularity: str, limit: int = 200) -> List[Candle]:
        """Mock candles."""
        return []
    
    def get_balances(self) -> List[Balance]:
        """Mock balances."""
        return []
    
    def get_positions(self) -> List[Position]:
        """Mock positions."""
        return []
    
    def is_connected(self) -> bool:
        """Always connected for testing."""
        return True
    
    def start_market_data(self, symbols: List[str]) -> None:
        """Mock WebSocket market data start."""
        self._ws_client = "mock_ws_client"  # Mock connection
        
        for symbol in symbols:
            # Initialize market data
            self._market_data[symbol] = {
                'mid': Decimal('50000') if symbol.startswith('BTC') else Decimal('3000'),
                'spread_bps': 10.0,  # 10 bps
                'depth_l1': Decimal('100000'),  # USD notional
                'depth_l10': Decimal('1000000'),  # USD notional
                'last_update': '2024-01-01T00:00:00Z'
            }
            
            # Initialize rolling windows
            self._rolling_windows[symbol] = {
                'vol_1m': 0.0,
                'vol_5m': 0.0
            }
        
        logger.info(f"Mock: Started market data for {symbols}")
    
    def is_stale(self, symbol: str, threshold_seconds: int = 10) -> bool:
        """Mock staleness check."""
        # Return False (not stale) if we have market data
        return symbol not in self._market_data
    
    def get_market_snapshot(self, symbol: str) -> Dict:
        """Mock market snapshot."""
        return {
            'vol_1m': 100.5,  # Mock volume
            'vol_5m': 500.0,
            'spread_bps': 10.0,
            'depth_l1': Decimal('100000'),  # USD notional
            'depth_l10': Decimal('1000000')  # USD notional
        }