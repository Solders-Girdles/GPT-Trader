#!/usr/bin/env python3
"""
Validation script for Product Catalog & Metadata.
Verifies product enrichment, catalog operations, and rule enforcement.
"""

import sys
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any

# Add src to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from bot_v2.features.brokerages.coinbase.models import to_product
from bot_v2.features.brokerages.coinbase.utils import (
    ProductCatalog,
    enforce_perp_rules,
    quantize_to_increment
)
from bot_v2.features.brokerages.core.interfaces import Product, MarketType
from bot_v2.features.brokerages.coinbase.errors import InvalidRequestError, NotFoundError


def validate_product_mapping():
    """Validate that to_product handles perps fields correctly."""
    print("\n=== Validating Product Mapping ===")
    
    # Test 1: Spot product (no perps fields)
    spot_payload = {
        "product_id": "BTC-USD",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "base_min_size": "0.001",
        "base_increment": "0.00001",
        "quote_increment": "0.01"
    }
    
    spot = to_product(spot_payload)
    assert spot.market_type == MarketType.SPOT, "Should identify as SPOT"
    assert spot.contract_size is None, "Spot should have no contract_size"
    assert spot.funding_rate is None, "Spot should have no funding_rate"
    assert spot.next_funding_time is None, "Spot should have no funding_time"
    print("✓ Spot product mapping correct")
    
    # Test 2: Perpetual with full metadata
    perp_payload = {
        "product_id": "BTC-PERP",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "contract_type": "perpetual",
        "base_min_size": "0.001",
        "base_increment": "0.00001",
        "quote_increment": "0.01",
        "max_leverage": 20,
        "contract_size": "1",
        "funding_rate": "0.0001",
        "next_funding_time": "2024-01-15T16:00:00Z"
    }
    
    perp = to_product(perp_payload)
    assert perp.market_type == MarketType.PERPETUAL, "Should identify as PERPETUAL"
    assert perp.contract_size == Decimal("1"), f"Contract size should be 1, got {perp.contract_size}"
    assert perp.funding_rate == Decimal("0.0001"), f"Funding rate should be 0.0001, got {perp.funding_rate}"
    assert perp.next_funding_time == datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc), "Funding time not parsed correctly"
    assert perp.leverage_max == 20, f"Leverage should be 20, got {perp.leverage_max}"
    print("✓ Perpetual product mapping with full metadata correct")
    
    # Test 3: Future product
    future_payload = {
        "product_id": "BTC-USD-240331",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "contract_type": "future",
        "base_min_size": "0.001",
        "base_increment": "0.00001",
        "quote_increment": "0.01",
        "expiry": "2024-03-31T08:00:00Z"
    }
    
    future = to_product(future_payload)
    assert future.market_type == MarketType.FUTURES, "Should identify as FUTURES"
    assert future.expiry == datetime(2024, 3, 31, 8, 0, 0, tzinfo=timezone.utc), "Expiry not parsed correctly"
    print("✓ Futures product mapping correct")
    
    print("✅ All product mappings validated")
    return True


def validate_product_catalog():
    """Validate ProductCatalog operations."""
    print("\n=== Validating ProductCatalog ===")
    
    # Create mock client
    class MockClient:
        def __init__(self):
            self.call_count = 0
        
        def get_products(self) -> Dict[str, Any]:
            self.call_count += 1
            return {
                "products": [
                    {
                        "product_id": "BTC-USD",
                        "base_currency": "BTC",
                        "quote_currency": "USD",
                        "base_min_size": "0.001",
                        "base_increment": "0.00001",
                        "quote_increment": "0.01"
                    },
                    {
                        "product_id": "BTC-PERP",
                        "base_currency": "BTC",
                        "quote_currency": "USD",
                        "contract_type": "perpetual",
                        "base_min_size": "0.001",
                        "base_increment": "0.00001",
                        "quote_increment": "0.01",
                        "contract_size": "1",
                        "funding_rate": "0.0001",
                        "next_funding_time": "2024-01-15T16:00:00Z",
                        "max_leverage": 20
                    }
                ]
            }
    
    client = MockClient()
    catalog = ProductCatalog(ttl_seconds=900)
    
    # Test 1: Get triggers refresh
    btc_usd = catalog.get(client, "BTC-USD")
    assert btc_usd.symbol == "BTC-USD", "Should get BTC-USD"
    assert btc_usd.market_type == MarketType.SPOT, "BTC-USD should be SPOT"
    assert client.call_count == 1, "Should have called get_products once"
    print("✓ Catalog get() triggers refresh")
    
    # Test 2: Get uses cache
    btc_usd_2 = catalog.get(client, "BTC-USD")
    assert client.call_count == 1, "Should still be 1 (used cache)"
    print("✓ Catalog uses cache for subsequent gets")
    
    # Test 3: Get perpetual with metadata
    btc_perp = catalog.get(client, "BTC-PERP")
    assert btc_perp.symbol == "BTC-PERP", "Should get BTC-PERP"
    assert btc_perp.market_type == MarketType.PERPETUAL, "Should be PERPETUAL"
    assert btc_perp.contract_size == Decimal("1"), "Should have contract_size"
    assert btc_perp.funding_rate == Decimal("0.0001"), "Should have funding_rate"
    assert btc_perp.leverage_max == 20, "Should have leverage_max"
    print("✓ Catalog returns perpetual with metadata")
    
    # Test 4: get_funding for perpetual
    funding_rate, next_funding = catalog.get_funding(client, "BTC-PERP")
    assert funding_rate == Decimal("0.0001"), f"Funding rate should be 0.0001, got {funding_rate}"
    assert next_funding == datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc), "Funding time incorrect"
    print("✓ get_funding() returns correct data for perpetual")
    
    # Test 5: get_funding for spot returns None
    funding_rate, next_funding = catalog.get_funding(client, "BTC-USD")
    assert funding_rate is None, "Spot should have no funding_rate"
    assert next_funding is None, "Spot should have no funding_time"
    print("✓ get_funding() returns None for spot")
    
    # Test 6: NotFoundError for missing product
    try:
        catalog.get(client, "MISSING-PERP")
        assert False, "Should have raised NotFoundError"
    except NotFoundError as e:
        assert "MISSING-PERP" in str(e), "Error should mention symbol"
        print("✓ NotFoundError raised for missing product")
    
    print("✅ ProductCatalog operations validated")
    return True


def validate_enforce_perp_rules():
    """Validate enforce_perp_rules helper."""
    print("\n=== Validating enforce_perp_rules ===")
    
    # Create test product
    product = Product(
        symbol="BTC-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.00001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
        leverage_max=20
    )
    
    # Test 1: Quantity quantization
    qty, price = enforce_perp_rules(product, Decimal("0.123456789"))
    assert qty == Decimal("0.12345"), f"Qty should be 0.12345, got {qty}"
    assert price is None, "Price should be None when not provided"
    print("✓ Quantity quantized to step_size")
    
    # Test 2: Price quantization
    qty, price = enforce_perp_rules(product, Decimal("0.01"), Decimal("50123.456"))
    assert qty == Decimal("0.01"), f"Qty should be 0.01, got {qty}"
    assert price == Decimal("50123.45"), f"Price should be 50123.45, got {price}"
    print("✓ Price quantized to price_increment")
    
    # Test 3: Min size validation
    try:
        enforce_perp_rules(product, Decimal("0.0001"))  # Below min_size
        assert False, "Should have raised InvalidRequestError"
    except InvalidRequestError as e:
        assert "below minimum size" in str(e), "Error should mention minimum size"
        print("✓ Rejects quantity below min_size")
    
    # Test 4: Min notional validation
    try:
        # 0.001 * 100 = 0.1 (below min_notional of 10)
        enforce_perp_rules(product, Decimal("0.001"), Decimal("100"))
        assert False, "Should have raised InvalidRequestError"
    except InvalidRequestError as e:
        assert "below minimum" in str(e), "Error should mention minimum notional"
        print("✓ Rejects order below min_notional")
    
    # Test 5: Valid order passes
    qty, price = enforce_perp_rules(product, Decimal("0.001"), Decimal("20000"))
    notional = qty * price
    assert notional >= product.min_notional, f"Notional {notional} should meet minimum"
    print("✓ Valid order passes all checks")
    
    print("✅ enforce_perp_rules validated")
    return True


def validate_quantize_to_increment():
    """Validate quantize_to_increment helper."""
    print("\n=== Validating quantize_to_increment ===")
    
    # Test 1: Basic quantization
    result = quantize_to_increment(Decimal("1.2345"), Decimal("0.01"))
    assert result == Decimal("1.23"), f"Should be 1.23, got {result}"
    print("✓ Basic quantization works")
    
    # Test 2: Floors, doesn't round
    result = quantize_to_increment(Decimal("1.2389"), Decimal("0.01"))
    assert result == Decimal("1.23"), f"Should floor to 1.23, got {result}"
    print("✓ Quantization floors (doesn't round)")
    
    # Test 3: Zero increment returns original
    result = quantize_to_increment(Decimal("1.2345"), Decimal("0"))
    assert result == Decimal("1.2345"), "Zero increment should return original"
    
    result = quantize_to_increment(Decimal("1.2345"), None)
    assert result == Decimal("1.2345"), "None increment should return original"
    print("✓ Zero/None increment handled")
    
    # Test 4: Non-power-of-10 increments
    result = quantize_to_increment(Decimal("1.237"), Decimal("0.025"))
    assert result == Decimal("1.225"), f"Should be 1.225 (49*0.025), got {result}"
    print("✓ Non-power-of-10 increments work")
    
    print("✅ quantize_to_increment validated")
    return True


def main():
    """Run all product catalog validations."""
    print("=" * 60)
    print("VALIDATION: Product Catalog & Metadata")
    print("=" * 60)
    
    try:
        # Run all validations
        results = [
            validate_product_mapping(),
            validate_product_catalog(),
            validate_enforce_perp_rules(),
            validate_quantize_to_increment()
        ]
        
        if all(results):
            print("\n" + "=" * 60)
            print("✅ VALIDATION SUCCESSFUL")
            print("=" * 60)
            print("\nSummary:")
            print("- Product model extended with perps fields")
            print("- to_product() maps all derivative metadata")
            print("- ProductCatalog caches and serves perps products")
            print("- get_funding() helper provides funding info")
            print("- enforce_perp_rules() validates and quantizes orders")
            print("- All rule enforcement working correctly")
            return 0
        else:
            print("\n❌ VALIDATION FAILED")
            return 1
            
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
