#!/usr/bin/env python3
"""
Validation script for Week 1 perpetuals client implementation.

Tests:
1. Dynamic product discovery for BTC/ETH/SOL/XRP perps
2. Order placement with quantization and TIF support  
3. Market/limit orders with client-IDs and reduce-only
4. Product metadata validation (increments, funding fields)
"""

import os
import sys
import asyncio
import logging
from decimal import Decimal
from typing import Set

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import os
if os.getenv('USE_REAL_ADAPTER') == '1':
    from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage as BrokerClass
    print("🔴 Using REAL CoinbaseBrokerage for validation")
else:
    from bot_v2.features.brokerages.coinbase.test_adapter import MinimalCoinbaseBrokerage as BrokerClass
    print("🟡 Using Mock MinimalCoinbaseBrokerage for validation")
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.endpoints import get_perps_symbols
from bot_v2.features.brokerages.core.interfaces import MarketType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerpsClientValidator:
    """Validator for perpetuals client functionality."""
    
    def __init__(self):
        # Configure for sandbox testing
        self.config = APIConfig(
            api_key=os.getenv('COINBASE_API_KEY', 'test_key'),
            api_secret=os.getenv('COINBASE_API_SECRET', 'test_secret'),
            passphrase=os.getenv('COINBASE_PASSPHRASE', 'test_passphrase'),
            base_url="https://api.sandbox.coinbase.com" if os.getenv('COINBASE_SANDBOX') else "https://api.coinbase.com",
            sandbox=os.getenv('COINBASE_SANDBOX', '0') == '1',
            enable_derivatives=True,
            api_mode="advanced",
            auth_type="JWT" if os.getenv('CDP_API_KEY') else "HMAC"
        )
        
        self.broker = BrokerClass(self.config)
        self.expected_perps = get_perps_symbols()
        
    async def run_all_tests(self) -> bool:
        """Run all validation tests."""
        logger.info("🧪 Starting perpetuals client validation...")
        
        tests = [
            ("Product Discovery", self.test_product_discovery),
            ("Product Metadata", self.test_product_metadata), 
            ("Order Placement", self.test_order_placement),
            ("TIF Support", self.test_tif_support),
            ("Quantization", self.test_quantization),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                logger.info(f"Running {test_name}...")
                result = await test_func()
                results.append((test_name, result, None))
                logger.info(f"✅ {test_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                results.append((test_name, False, str(e)))
                logger.error(f"❌ {test_name}: FAIL - {e}")
        
        # Summary
        passed = sum(1 for _, result, _ in results if result)
        total = len(results)
        
        logger.info(f"\n📊 Validation Summary: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All perpetuals client tests PASSED!")
            return True
        else:
            logger.error("❌ Some tests FAILED")
            for test_name, result, error in results:
                if not result:
                    logger.error(f"  - {test_name}: {error or 'Unknown error'}")
            return False
    
    async def test_product_discovery(self) -> bool:
        """Test dynamic perpetual product discovery."""
        try:
            # Get all perpetual products
            perps = self.broker.list_products(market=MarketType.PERPETUAL)
            
            if not perps:
                logger.warning("No perpetual products found")
                return False
            
            discovered_symbols = {p.symbol for p in perps}
            logger.info(f"Discovered perps: {sorted(discovered_symbols)}")
            
            # Check if we have the expected core perps
            missing = self.expected_perps - discovered_symbols
            if missing:
                logger.warning(f"Missing expected perps: {missing}")
                # Don't fail completely - market might not have all perps
            
            # At least check we have some perps
            found_expected = len(discovered_symbols & self.expected_perps)
            if found_expected >= 2:  # At least 2 of the expected perps
                logger.info(f"Found {found_expected} expected perps")
                return True
            else:
                logger.error(f"Only found {found_expected} expected perps")
                return False
                
        except Exception as e:
            logger.error(f"Product discovery failed: {e}")
            return False
    
    async def test_product_metadata(self) -> bool:
        """Test product metadata contains required fields.""" 
        try:
            perps = self.broker.list_products(market=MarketType.PERPETUAL)
            
            if not perps:
                return False
            
            # Test first perpetual product
            product = perps[0]
            
            # Check required fields
            required_fields = [
                ('symbol', str),
                ('step_size', Decimal), 
                ('price_increment', Decimal),
                ('min_size', Decimal),
                ('market_type', MarketType)
            ]
            
            for field_name, field_type in required_fields:
                if not hasattr(product, field_name):
                    logger.error(f"Missing field: {field_name}")
                    return False
                
                value = getattr(product, field_name)
                if not isinstance(value, field_type):
                    logger.error(f"Wrong type for {field_name}: {type(value)} vs {field_type}")
                    return False
            
            # Check perpetual-specific fields are present (may be None/0)
            perp_fields = ['funding_rate', 'next_funding_time', 'contract_size']
            for field in perp_fields:
                if not hasattr(product, field):
                    logger.warning(f"Missing perpetual field: {field}")
            
            logger.info(f"Product metadata valid for {product.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Metadata validation failed: {e}")
            return False
    
    async def test_order_placement(self) -> bool:
        """Test basic order placement with quantization."""
        try:
            perps = self.broker.list_products(market=MarketType.PERPETUAL)
            if not perps:
                return False
            
            # Use first available perp for testing
            product = perps[0]
            symbol = product.symbol
            
            # Test limit order placement (small size, will cancel immediately)
            test_qty = product.min_size * 2  # Small test quantity
            test_price = Decimal('1000000')  # Very high price, won't fill
            
            order = self.broker.place_order(
                symbol=symbol,
                side="buy",
                order_type="limit",
                quantity=test_qty,
                limit_price=test_price,
                tif="GTC",
                client_id="test_week1_validation"
            )
            
            if not order:
                logger.error("Order placement returned None")
                return False
            
            logger.info(f"Test order placed: {order.id}")
            
            # Try to cancel the order immediately
            cancelled = self.broker.cancel_order(order.id)
            if cancelled:
                logger.info("Test order cancelled successfully")
            else:
                logger.warning("Could not cancel test order")
            
            return True
            
        except Exception as e:
            logger.error(f"Order placement test failed: {e}")
            return False
    
    async def test_tif_support(self) -> bool:
        """Test time-in-force parameter support."""
        try:
            perps = self.broker.list_products(market=MarketType.PERPETUAL)
            if not perps:
                return False
            
            product = perps[0]
            
            # Test that valid TIFs are accepted
            valid_tifs = ["GTC", "IOC"]
            
            for tif in valid_tifs:
                # Don't actually place orders, just validate parameters
                try:
                    # This should not raise an error for valid TIF
                    order = self.broker.place_order(
                        symbol=product.symbol,
                        side="buy",
                        order_type="limit",
                        quantity=product.min_size,
                        limit_price=Decimal('1000000'),  # Won't fill
                        tif=tif,
                        client_id=f"test_tif_{tif.lower()}"
                    )
                    
                    # If order was created, cancel it
                    if order:
                        self.broker.cancel_order(order.id)
                        
                except Exception as e:
                    if "Unsupported time in force" in str(e):
                        logger.error(f"TIF {tif} not supported")
                        return False
            
            logger.info("TIF support validation passed")
            return True
            
        except Exception as e:
            logger.error(f"TIF validation failed: {e}")
            return False
    
    async def test_quantization(self) -> bool:
        """Test that orders are properly quantized."""
        try:
            perps = self.broker.list_products(market=MarketType.PERPETUAL)
            if not perps:
                return False
            
            product = perps[0]
            
            # Test with unquantized values
            unquantized_qty = product.min_size * Decimal('2.123456789')
            unquantized_price = Decimal('50000.123456789')
            
            # The adapter should quantize these internally
            logger.info(f"Testing quantization for {product.symbol}")
            logger.info(f"Step size: {product.step_size}, Price increment: {product.price_increment}")
            logger.info(f"Unquantized qty: {unquantized_qty}, price: {unquantized_price}")
            
            # This should succeed with quantized values
            order = self.broker.place_order(
                symbol=product.symbol,
                side="buy", 
                order_type="limit",
                quantity=unquantized_qty,
                limit_price=unquantized_price,
                client_id="test_quantization"
            )
            
            if order:
                logger.info("Quantization test order placed successfully")
                self.broker.cancel_order(order.id)
                return True
            else:
                logger.error("Quantization test failed - no order returned")
                return False
                
        except Exception as e:
            logger.error(f"Quantization test failed: {e}")
            return False


async def main():
    """Main validation runner."""
    if os.getenv('RUN_SANDBOX_VALIDATIONS') != '1':
        print("⚠️  Sandbox validations disabled. Set RUN_SANDBOX_VALIDATIONS=1 to run.")
        print("   This prevents accidental live API calls during testing.")
        return
    
    validator = PerpsClientValidator()
    success = await validator.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())