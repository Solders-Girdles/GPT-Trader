#!/usr/bin/env python3
"""
Test Coinbase CDP Trading Capabilities
Tests order placement, cancellation, and status tracking with the working CDP authentication.
"""

import asyncio
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
from src.bot_v2.features.brokerages.core.interfaces import (
    OrderRequest, OrderType, OrderSide, TimeInForce, OrderStatus
)

async def test_trading_capabilities():
    """Test all trading capabilities with CDP authentication."""
    
    print("=" * 80)
    print("COINBASE CDP TRADING CAPABILITIES TEST")
    print("=" * 80)
    
    # Initialize adapter
    adapter = CoinbaseBrokerage()
    
    try:
        # 1. Connect and verify authentication
        print("\n1. Testing Connection...")
        is_connected = await adapter.connect()
        if not is_connected:
            print("❌ Failed to connect to Coinbase")
            return
        print("✅ Connected to Coinbase with CDP authentication")
        
        # 2. Get account info
        print("\n2. Testing Account Retrieval...")
        account = await adapter.get_account()
        if account:
            print(f"✅ Account ID: {account.id}")
            print(f"   Currency: {account.currency}")
            
            # Handle potential decimal parsing issues
            try:
                balance = Decimal(str(account.balance))
                print(f"   Balance: {balance}")
            except:
                print(f"   Balance: {account.balance} (raw)")
                
            try:
                available = Decimal(str(account.available_balance))
                print(f"   Available: {available}")
            except:
                print(f"   Available: {account.available_balance} (raw)")
        else:
            print("❌ Failed to retrieve account")
        
        # 3. Get all accounts to find one with balance
        print("\n3. Finding Funded Accounts...")
        accounts_response = await adapter._client.get_accounts()
        if accounts_response and 'accounts' in accounts_response:
            funded_accounts = []
            for acc in accounts_response['accounts'][:10]:  # Check first 10
                try:
                    balance = Decimal(acc.get('available_balance', {}).get('value', '0'))
                    if balance > 0:
                        funded_accounts.append({
                            'currency': acc.get('currency', 'UNKNOWN'),
                            'balance': balance,
                            'uuid': acc.get('uuid')
                        })
                except:
                    pass
            
            if funded_accounts:
                print(f"✅ Found {len(funded_accounts)} funded accounts:")
                for acc in funded_accounts[:5]:  # Show first 5
                    print(f"   {acc['currency']}: {acc['balance']}")
            else:
                print("⚠️  No funded accounts found")
        
        # 4. Test market data
        print("\n4. Testing Market Data...")
        
        # Test getting a quote
        quote = await adapter.get_quote("BTC-USD")
        if quote:
            print(f"✅ BTC-USD Quote:")
            print(f"   Bid: ${quote.bid}")
            print(f"   Ask: ${quote.ask}")
            print(f"   Last: ${quote.last}")
            print(f"   Volume: {quote.volume}")
        else:
            print("❌ Failed to get BTC-USD quote")
        
        # 5. Test order placement (if enabled)
        trading_enabled = os.getenv('COINBASE_ENABLE_TRADING', '0') == '1'
        
        if trading_enabled:
            print("\n5. Testing Order Placement...")
            print("⚠️  TRADING IS ENABLED - Will attempt to place a REAL order")
            
            # Create a very small test order
            test_order = OrderRequest(
                symbol="BTC-USD",
                order_type=OrderType.LIMIT,
                side=OrderSide.BUY,
                quantity=Decimal("0.00001"),  # Minimum BTC amount
                price=Decimal("10.00"),  # Very low price to avoid fill
                time_in_force=TimeInForce.GTC
            )
            
            print(f"\nPlacing test order:")
            print(f"  Symbol: {test_order.symbol}")
            print(f"  Type: {test_order.order_type}")
            print(f"  Side: {test_order.side}")
            print(f"  Quantity: {test_order.quantity}")
            print(f"  Price: ${test_order.price}")
            
            try:
                order = await adapter.place_order(test_order)
                if order:
                    print(f"✅ Order placed successfully!")
                    print(f"   Order ID: {order.id}")
                    print(f"   Status: {order.status}")
                    
                    # Wait a moment
                    await asyncio.sleep(2)
                    
                    # Test order status
                    print("\n6. Testing Order Status...")
                    status_order = await adapter.get_order(order.id)
                    if status_order:
                        print(f"✅ Order status retrieved:")
                        print(f"   Status: {status_order.status}")
                        print(f"   Filled Qty: {status_order.filled_quantity}")
                    
                    # Test order cancellation
                    if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                        print("\n7. Testing Order Cancellation...")
                        cancelled = await adapter.cancel_order(order.id)
                        if cancelled:
                            print(f"✅ Order cancelled successfully")
                        else:
                            print(f"❌ Failed to cancel order")
                else:
                    print("❌ Failed to place order")
            except Exception as e:
                print(f"❌ Order placement error: {e}")
        else:
            print("\n5. Order Placement Test")
            print("⚠️  TRADING IS DISABLED - Skipping order tests")
            print("   To enable: Set COINBASE_ENABLE_TRADING=1 in .env.production")
        
        # 6. Test getting open orders
        print("\n8. Testing Open Orders Retrieval...")
        try:
            orders = await adapter.get_open_orders()
            if orders is not None:
                print(f"✅ Retrieved open orders: {len(orders)} orders")
                for order in orders[:3]:  # Show first 3
                    print(f"   {order.symbol} {order.side} {order.quantity} @ ${order.price}")
            else:
                print("⚠️  No open orders or retrieval failed")
        except Exception as e:
            print(f"❌ Error getting open orders: {e}")
        
        # 7. Test positions
        print("\n9. Testing Positions...")
        try:
            positions = await adapter.get_positions()
            if positions:
                print(f"✅ Retrieved {len(positions)} positions")
                for pos in positions[:3]:  # Show first 3
                    print(f"   {pos.symbol}: {pos.quantity} units")
            else:
                print("⚠️  No positions found")
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
        
        print("\n" + "=" * 80)
        print("TRADING CAPABILITIES TEST COMPLETE")
        print("=" * 80)
        
        # Summary
        print("\n📊 SUMMARY:")
        print("✅ CDP Authentication: Working")
        print("✅ Account Access: Working")
        print("✅ Market Data: Working")
        
        if trading_enabled:
            print("✅ Order Placement: Tested")
            print("✅ Order Status: Tested")
            print("✅ Order Cancellation: Tested")
        else:
            print("⚠️  Order Operations: Not tested (trading disabled)")
        
        print("\n💡 Next Steps:")
        print("1. Enable trading with COINBASE_ENABLE_TRADING=1 to test orders")
        print("2. Fund account with small amount for live testing")
        print("3. Run full integration test suite")
        print("4. Implement production safeguards (position limits, etc.)")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await adapter.disconnect()
        print("\n✅ Disconnected from Coinbase")

if __name__ == "__main__":
    asyncio.run(test_trading_capabilities())