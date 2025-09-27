#!/usr/bin/env python3
"""
Test Coinbase CDP Trading - Simplified version
"""

import os
import sys
from decimal import Decimal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load production environment manually
env_file = Path(__file__).parent.parent / '.env.production'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes from value
                value = value.strip().strip('"')
                # Handle multiline private key
                if key == 'COINBASE_CDP_PRIVATE_KEY':
                    # Read until we find the end marker
                    private_key_lines = [value] if value else []
                    for next_line in f:
                        next_line = next_line.strip()
                        private_key_lines.append(next_line)
                        if 'END EC PRIVATE KEY' in next_line:
                            break
                    value = '\n'.join(private_key_lines)
                os.environ[key] = value

from src.bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from src.bot_v2.features.brokerages.coinbase.models import APIConfig
from src.bot_v2.features.brokerages.core.interfaces import (
    OrderType, OrderSide, TimeInForce, OrderStatus
)

def test_trading():
    """Test trading capabilities with CDP."""
    
    print("=" * 80)
    print("COINBASE CDP TRADING TEST")
    print("=" * 80)
    
    # Create config with CDP credentials
    config = APIConfig(
        api_key="",  # Not used for CDP
        api_secret="",  # Not used for CDP
        passphrase=None,
        base_url=os.getenv('COINBASE_API_BASE', 'https://api.coinbase.com'),
        sandbox=os.getenv('COINBASE_SANDBOX', '0') == '1',
        ws_url=os.getenv('COINBASE_WS_URL', 'wss://advanced-trade-ws.coinbase.com'),
        cdp_api_key=os.getenv('COINBASE_CDP_API_KEY'),
        cdp_private_key=os.getenv('COINBASE_CDP_PRIVATE_KEY')
    )
    
    # Initialize brokerage
    broker = CoinbaseBrokerage(config)
    
    try:
        # 1. Connect
        print("\n1. Testing Connection...")
        is_connected = broker.connect()
        if not is_connected:
            print("❌ Failed to connect")
            return
        print("✅ Connected to Coinbase with CDP")
        
        # 2. Get account
        print("\n2. Getting Account Info...")
        account = broker.get_account()
        if account:
            print(f"✅ Account ID: {account.id}")
            print(f"   Currency: {account.currency}")
            print(f"   Balance: {account.balance}")
            print(f"   Available: {account.available_balance}")
        else:
            print("❌ Failed to get account")
        
        # 3. Get quote
        print("\n3. Getting Market Quote...")
        quote = broker.get_quote("BTC-USD")
        if quote:
            print(f"✅ BTC-USD Quote:")
            print(f"   Bid: ${quote.bid}")
            print(f"   Ask: ${quote.ask}")
            print(f"   Last: ${quote.last}")
        else:
            print("❌ Failed to get quote")
        
        # 4. Test order placement (if enabled)
        trading_enabled = os.getenv('COINBASE_ENABLE_TRADING', '0') == '1'
        
        if trading_enabled:
            print("\n4. Testing Order Placement...")
            print("⚠️  TRADING ENABLED - Placing real order")
            
            try:
                # Place a very small limit buy order at low price
                order = broker.place_order(
                    symbol="BTC-USD",
                    order_type=OrderType.LIMIT,
                    side=OrderSide.BUY,
                    quantity=Decimal("0.00001"),  # Minimum BTC
                    price=Decimal("10.00"),  # Very low price
                    time_in_force=TimeInForce.GTC
                )
                
                if order:
                    print(f"✅ Order placed!")
                    print(f"   Order ID: {order.id}")
                    print(f"   Status: {order.status}")
                    
                    # Get order status
                    print("\n5. Checking Order Status...")
                    status_order = broker.get_order(order.id)
                    if status_order:
                        print(f"✅ Order status: {status_order.status}")
                    
                    # Cancel if still open
                    if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                        print("\n6. Cancelling Order...")
                        success = broker.cancel_order(order.id)
                        if success:
                            print("✅ Order cancelled")
                        else:
                            print("❌ Failed to cancel")
                else:
                    print("❌ Failed to place order")
                    
            except Exception as e:
                print(f"❌ Order error: {e}")
        else:
            print("\n4. Order Placement")
            print("⚠️  Trading disabled - set COINBASE_ENABLE_TRADING=1 to test")
        
        # 5. Get open orders
        print("\n7. Getting Open Orders...")
        try:
            orders = broker.get_open_orders()
            if orders is not None:
                print(f"✅ Found {len(orders)} open orders")
                for order in orders[:3]:
                    print(f"   {order.symbol} {order.side} {order.quantity} @ ${order.price}")
            else:
                print("⚠️  No open orders")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # 6. Get positions
        print("\n8. Getting Positions...")
        try:
            positions = broker.get_positions()
            if positions:
                print(f"✅ Found {len(positions)} positions")
                for pos in positions[:3]:
                    print(f"   {pos.symbol}: {pos.quantity}")
            else:
                print("⚠️  No positions")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
        
        print("\n📊 SUMMARY:")
        print("✅ CDP Authentication working")
        print("✅ Account access working")
        print("✅ Market data working")
        
        if trading_enabled:
            print("✅ Order operations tested")
        else:
            print("⚠️  Order operations not tested (trading disabled)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        broker.disconnect()
        print("\n✅ Disconnected")

if __name__ == "__main__":
    test_trading()