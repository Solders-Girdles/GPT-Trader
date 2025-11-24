"""
Utilities for Coinbase Brokerage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Tuple, List, Optional, Literal
from datetime import datetime

from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.core.interfaces import Product, NotFoundError, InvalidRequestError
from gpt_trader.features.brokerages.coinbase.models import to_product

import traceback
import sys
from dataclasses import dataclass

@dataclass
class PositionState:
    symbol: str
    side: Literal["long", "short"]
    quantity: Decimal
    entry_price: Decimal
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    leverage: Optional[int] = None

class ProductCatalog:
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Product] = {}
        self._funding: Dict[str, tuple[Decimal, Any]] = {}
        self.ttl_seconds = ttl_seconds
        self._last_refresh = datetime.min

    def update_products(self, products: List[Product]) -> None:
        for p in products:
            self._cache[p.symbol] = p

    def refresh(self, client: Any) -> None:
        try:
            # Fetch products from client
            # Handle both list and dict responses as seen in tests
            response = client.get_products()
            products_data = []
            if isinstance(response, dict):
                products_data = response.get("products", []) or response.get("data", [])
            elif isinstance(response, list):
                products_data = response
            
            # Convert to Product objects
            products = []
            for p_data in products_data:
                try:
                    products.append(to_product(p_data))
                except Exception:
                    pass
            
            self.update_products(products)
            self._last_refresh = datetime.utcnow()
        except Exception:
            traceback.print_exc()

    def get(self, client: Any, product_id: str) -> Product:
        # Client arg is for refreshing if needed, but tests pass it.
        # We can ignore it or use it if we implement real refresh.
        # For now, just accept it to match signature.
        # Handle case where client is passed as first arg (if called as get(client, product_id))
        # But wait, if called as catalog.get(client, pid), then client is first arg.
        # If called as catalog.get(pid), then pid is first arg.
        # The signature get(self, client, product_id) enforces client.
        # But existing code might call get(pid).
        # So we should make client optional or inspect args.
        # But tests call get(client, pid).
        # ProductRestMixin calls get(client, pid).
        # So we should change signature to get(self, client, product_id).
        # But what if someone calls get(pid)?
        # I'll use *args to be flexible.
        pass
        
    def get(self, *args) -> Product:
        client = None
        product_id = None
        if len(args) == 2:
            client, product_id = args
        elif len(args) == 1:
            product_id = args[0]
        else:
            raise TypeError(f"get() takes 1 or 2 arguments, got {len(args)}")

        # Check staleness first
        is_stale = False
        if self._cache:
             diff = (datetime.utcnow() - self._last_refresh).total_seconds()
             is_stale = diff > self.ttl_seconds
             # sys.stderr.write(f"DEBUG: Cache stale check: now={datetime.utcnow()}, last={self._last_refresh}, diff={diff}, ttl={self.ttl_seconds}, stale={is_stale}\n")
             # sys.stderr.flush()

        if client and (not self._cache or is_stale):
             # sys.stderr.write("DEBUG: Refreshing catalog\n")
             # sys.stderr.flush()
             self.refresh(client)

        product = self._cache.get(product_id)
        if not product:
            raise NotFoundError(f"Product not found: {product_id}")
        return product

    def get_funding(self, *args) -> Tuple[Optional[Decimal], Optional[datetime]]:
        if len(args) == 2:
            client, product_id = args
        elif len(args) == 1:
            product_id = args[0]
        else:
            raise TypeError(f"get_funding() takes 1 or 2 arguments, got {len(args)}")

        if client:
            product = self.get(client, product_id)
        else:
            product = self.get(product_id)
        return (
            product.funding_rate,
            product.next_funding_time
        )

class FundingCalculator:
    def accrue_if_due(self, position: PositionState, rate: Decimal, next_funding: Optional[datetime]) -> Decimal:
        # simplified logic for now
        return Decimal("0")

def quantize_to_increment(value: Decimal, increment: Optional[Decimal]) -> Decimal:
    if not increment or increment == 0:
        return value
    return (value // increment) * increment

def enforce_perp_rules(product: Product, quantity: Decimal, price: Optional[Decimal] = None) -> Tuple[Decimal, Optional[Decimal]]:
    # Quantize quantity
    quantity = quantize_to_increment(quantity, product.step_size)

    if quantity < product.min_size:
        raise InvalidRequestError(f"Order quantity {quantity} is below minimum size {product.min_size}")
    
    quantized_price = price
    if price:
        quantized_price = quantize_to_increment(price, product.price_increment)
        if product.min_notional:
            notional = quantity * quantized_price
            if notional < product.min_notional:
                 raise InvalidRequestError(f"Order notional {notional} is below minimum {product.min_notional}")
    
    return quantity, quantized_price