#!/usr/bin/env python3
"""
Test script to verify account capabilities and product availability.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_v2.orchestration.broker_factory import create_brokerage


def test_account_capabilities():
    """Test what products and capabilities are available."""
    
    print("=" * 70)
    print("TESTING ACCOUNT CAPABILITIES")
    print("=" * 70)
    
    try:
        # Create and connect to broker
        broker = create_brokerage()
        
        print("\n1. Testing Connection...")
        if broker.connect():
            print("✅ Connected successfully")
        else:
            print("❌ Failed to connect")
            return
        
        print("\n2. Fetching Products...")
        products = broker.list_products()
        print(f"   Total products found: {len(products)}")
        
        # Count by type
        spot_count = 0
        perp_count = 0
        futures_count = 0
        
        perp_symbols = []
        
        for product in products:
            if hasattr(product, 'market_type'):
                market_type = str(product.market_type)
                if 'SPOT' in market_type:
                    spot_count += 1
                elif 'PERPETUAL' in market_type or 'PERP' in market_type:
                    perp_count += 1
                    perp_symbols.append(product.symbol)
                elif 'FUTURE' in market_type:
                    futures_count += 1
        
        print(f"\n   Product breakdown:")
        print(f"   - Spot products: {spot_count}")
        print(f"   - Perpetual products: {perp_count}")
        print(f"   - Futures products: {futures_count}")
        
        if perp_symbols:
            print(f"\n   Available perpetuals:")
            for symbol in perp_symbols[:10]:  # Show first 10
                print(f"   - {symbol}")
            if len(perp_symbols) > 10:
                print(f"   ... and {len(perp_symbols) - 10} more")
        
        print("\n3. Testing Account Info...")
        try:
            balances = broker.list_balances()
            print(f"   Found {len(balances)} currency balances")
            
            # Show USD-like balances
            for balance in balances:
                if balance.asset in ['USD', 'USDC', 'USDT']:
                    print(f"   - {balance.asset}: Available={balance.available}, Total={balance.total}")
        except Exception as e:
            print(f"   ⚠️  Could not fetch balances: {e}")
        
        print("\n4. Testing Derivatives Access...")
        # Try to get a perpetual product directly
        test_symbols = ['BTC-PERP', 'ETH-PERP', 'BTC-PERPETUAL', 'ETH-PERPETUAL']
        found_any = False
        
        for symbol in test_symbols:
            try:
                product = broker.get_product(symbol)
                if product:
                    print(f"   ✅ Found {symbol}: {product.market_type}")
                    found_any = True
                    break
            except Exception:
                continue
        
        if not found_any:
            print("   ❌ No perpetual products accessible")
            print("\n   POSSIBLE REASONS:")
            print("   1. Your account may not have derivatives trading enabled")
            print("   2. You may need to complete additional verification on Coinbase")
            print("   3. Derivatives may not be available in your region")
            print("\n   ACTION REQUIRED:")
            print("   - Log into Coinbase Advanced Trade")
            print("   - Check if you see 'Futures' or 'Perpetuals' in the product list")
            print("   - If not, look for 'Enable Derivatives' or contact support")
        
        print("\n5. Testing Orders Endpoint...")
        try:
            # Try to list orders (even if empty)
            orders = broker.list_orders()
            print(f"   ✅ Orders endpoint works (found {len(orders)} orders)")
        except Exception as e:
            print(f"   ❌ Orders endpoint failed: {e}")
            print("      This might be normal if you've never placed orders")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if perp_count > 0:
        print("✅ Your account HAS access to perpetuals!")
        print("   You can start trading with the canary profile")
    else:
        print("❌ Your account does NOT have perpetuals access")
        print("\n   Next steps:")
        print("   1. Log into Coinbase Advanced Trade")
        print("   2. Look for derivatives/futures enablement")
        print("   3. Complete any required verification")
        print("   4. Contact Coinbase support if needed")
        print("\n   For now, you can still:")
        print("   - Use --profile dev for mock trading")
        print("   - Trade spot markets if you modify the strategies")


if __name__ == "__main__":
    test_account_capabilities()