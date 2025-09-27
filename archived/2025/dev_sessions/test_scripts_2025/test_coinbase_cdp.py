#!/usr/bin/env python3
"""
Test Coinbase CDP JWT authentication and API integration.

This script tests the CDP authentication implementation with real credentials.
"""

import os
import sys
import logging
from decimal import Decimal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bot_v2.orchestration.broker_factory import create_brokerage
from src.bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_production_env():
    """Load production environment variables."""
    env_file = Path(__file__).parent.parent / ".env.production"
    if env_file.exists():
        logger.info(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Handle multi-line values (like private keys)
                        if value.startswith('"') and not value.endswith('"'):
                            # Multi-line value
                            lines = [value[1:]]  # Remove opening quote
                            for next_line in f:
                                next_line = next_line.strip()
                                if next_line.endswith('"'):
                                    lines.append(next_line[:-1])  # Remove closing quote
                                    value = '\n'.join(lines)
                                    break
                                lines.append(next_line)
                        else:
                            # Single line value - remove quotes if present
                            value = value.strip('"')
                        os.environ[key] = value
                        if key.startswith("COINBASE_CDP"):
                            logger.info(f"Set {key}=...{value[-20:] if len(value) > 20 else value}")
    else:
        logger.warning(f"No .env.production file found at {env_file}")


def test_connection():
    """Test basic connection and authentication."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Connection and Authentication")
    logger.info("=" * 60)
    
    try:
        broker = create_brokerage()
        logger.info("✅ Broker created successfully")
        
        # Test connection
        connected = broker.connect()
        if connected:
            logger.info("✅ Successfully connected to Coinbase")
            account_id = broker.get_account_id()
            logger.info(f"✅ Account ID: {account_id}")
            return broker
        else:
            logger.error("❌ Failed to connect to Coinbase")
            return None
            
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_account_info(broker):
    """Test account information retrieval."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Account Information")
    logger.info("=" * 60)
    
    try:
        # Get balances
        balances = broker.list_balances()
        logger.info(f"✅ Retrieved {len(balances)} balances")
        
        # Show non-zero balances
        for balance in balances:
            if balance.total > 0:
                logger.info(f"   {balance.asset}: Total={balance.total}, Available={balance.available}, Hold={balance.hold}")
        
        if not any(b.total > 0 for b in balances):
            logger.info("   (No non-zero balances)")
            
    except Exception as e:
        logger.error(f"❌ Failed to get account info: {e}")


def test_market_data(broker):
    """Test market data endpoints."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Market Data")
    logger.info("=" * 60)
    
    try:
        # List products
        products = broker.list_products()
        logger.info(f"✅ Retrieved {len(products)} products")
        
        # Test with BTC-USD
        symbol = "BTC-USD"
        
        # Get quote
        quote = broker.get_quote(symbol)
        logger.info(f"✅ {symbol} Quote: Bid=${quote.bid}, Ask=${quote.ask}, Last=${quote.last}")
        
        # Get candles
        candles = broker.get_candles(symbol, "ONE_MINUTE", limit=5)
        logger.info(f"✅ Retrieved {len(candles)} candles for {symbol}")
        if candles:
            latest = candles[0]
            logger.info(f"   Latest: O={latest.open}, H={latest.high}, L={latest.low}, C={latest.close}, V={latest.volume}")
            
    except Exception as e:
        logger.error(f"❌ Failed to get market data: {e}")


def test_order_management(broker):
    """Test order management (list only, no placement unless enabled)."""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Order Management")
    logger.info("=" * 60)
    
    try:
        # List recent orders
        orders = broker.list_orders()
        logger.info(f"✅ Retrieved {len(orders)} orders")
        
        # Show recent orders
        for order in orders[:3]:
            logger.info(f"   {order.order_id}: {order.symbol} {order.side.value} {order.qty} @ {order.price or 'market'} - {order.status.value}")
        
        # Check if trading is enabled
        if os.getenv("COINBASE_ENABLE_TRADING", "0") == "1":
            logger.info("\n⚠️  Trading is ENABLED - would place test order here")
            # Uncomment to actually place a test order:
            # test_place_order(broker)
        else:
            logger.info("\n✅ Trading is disabled (safe mode)")
            
    except Exception as e:
        logger.error(f"❌ Failed to test order management: {e}")


def test_place_order(broker):
    """Place a small test order (only if explicitly enabled)."""
    logger.info("\nPlacing test order...")
    
    symbol = os.getenv("COINBASE_ORDER_SYMBOL", "BTC-USD")
    qty = Decimal(os.getenv("COINBASE_TEST_QTY", "0.00001"))
    
    # Get current price for limit order
    quote = broker.get_quote(symbol)
    # Place limit buy well below market (won't fill)
    limit_price = quote.bid * Decimal("0.5")  # 50% below bid
    
    try:
        order = broker.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=qty,
            price=limit_price,
            tif=TimeInForce.GTC
        )
        logger.info(f"✅ Test order placed: {order.order_id}")
        
        # Cancel immediately
        if broker.cancel_order(order.order_id):
            logger.info(f"✅ Test order cancelled")
        else:
            logger.warning(f"⚠️  Could not cancel test order")
            
    except Exception as e:
        logger.error(f"❌ Failed to place test order: {e}")


def main():
    """Run all tests."""
    logger.info("Coinbase CDP Integration Test")
    logger.info("=" * 60)
    
    # Load production environment
    load_production_env()
    
    # Verify CDP credentials are set
    cdp_key = os.getenv("COINBASE_CDP_API_KEY")
    cdp_private = os.getenv("COINBASE_CDP_PRIVATE_KEY")
    
    if not cdp_key or not cdp_private:
        logger.error("❌ CDP credentials not found in environment")
        logger.error("   Please set COINBASE_CDP_API_KEY and COINBASE_CDP_PRIVATE_KEY")
        return
    
    logger.info(f"CDP API Key: ...{cdp_key[-30:] if len(cdp_key) > 30 else cdp_key}")
    logger.info(f"CDP Private Key: {'Present' if cdp_private else 'Missing'}")
    logger.info(f"Auth Type: {os.getenv('COINBASE_AUTH_TYPE', 'AUTO')}")
    logger.info(f"Environment: {'Sandbox' if os.getenv('COINBASE_SANDBOX') == '1' else 'Production'}")
    
    # Test connection
    broker = test_connection()
    if not broker:
        logger.error("\n❌ Connection failed - cannot continue tests")
        return
    
    # Run other tests
    test_account_info(broker)
    test_market_data(broker)
    test_order_management(broker)
    
    # Disconnect
    broker.disconnect()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All tests completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()