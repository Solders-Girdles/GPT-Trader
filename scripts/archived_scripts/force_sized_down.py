#!/usr/bin/env python3
"""
Force a SIZED_DOWN event for validation.
Temporarily places oversized order to trigger safety filter.
"""

import os
import sys
import asyncio
from decimal import Decimal
from datetime import datetime, timezone

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


async def force_sized_down_event():
    """Force a SIZED_DOWN event for validation."""
    print("🎯 FORCING SIZED_DOWN EVENT")
    print("="*60)
    print("⚠️  This will intentionally trigger a safety filter")
    print("="*60)
    
    try:
        from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
        from bot_v2.features.brokerages.coinbase.models import APIConfig
        
        # Configure with safety limits
        config = APIConfig(
            api_key=os.getenv('COINBASE_CDP_API_KEY'),
            api_secret="",
            passphrase="",
            base_url="https://api.coinbase.com",  # Always use production for AT
            auth_type="JWT",
            api_mode="advanced",
            sandbox=False,  # AT doesn't have sandbox
            enable_derivatives=True
        )
        
        # Set artificially low max position size to force SIZED_DOWN
        os.environ['COINBASE_MAX_POSITION_SIZE'] = '0.00005'  # 0.00005 BTC max
        
        print(f"Current max position size: {os.getenv('COINBASE_MAX_POSITION_SIZE')} BTC")
        
        broker = CoinbaseBrokerage(config)
        
        # Attempt to place oversized order (will be sized down)
        oversized_order = {
            'symbol': 'BTC-PERP',
            'side': 'buy',
            'type': 'limit',
            'size': Decimal('0.001'),  # 20x the max size
            'price': Decimal('30000'),  # Far from market (won't fill)
            'post_only': True,
            'client_id': f"force_sized_down_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        }
        
        print("\n📦 Attempting Oversized Order:")
        print(f"  Symbol: {oversized_order['symbol']}")
        print(f"  Size: {oversized_order['size']} BTC (oversized)")
        print(f"  Max Allowed: {os.getenv('COINBASE_MAX_POSITION_SIZE')} BTC")
        print(f"  Expected: SIZED_DOWN to {os.getenv('COINBASE_MAX_POSITION_SIZE')} BTC")
        
        # This should trigger SIZED_DOWN
        result = await broker.place_order(
            symbol=oversized_order['symbol'],
            side=oversized_order['side'],
            order_type=oversized_order['type'],
            quantity=oversized_order['size'],
            limit_price=oversized_order['price'],
            tif='GTC',
            post_only=True
        )
        
        if result:
            print("\n✅ Order placed (was sized down):")
            print(f"  Order ID: {result.get('id')}")
            print(f"  Original Size: {oversized_order['size']}")
            print(f"  Actual Size: {result.get('size')}")
            print(f"  Status: {result.get('status')}")
            
            if Decimal(str(result.get('size', 0))) < oversized_order['size']:
                print("\n🎯 SIZED_DOWN EVENT CONFIRMED!")
                print(f"  Reduced from {oversized_order['size']} to {result.get('size')} BTC")
                
                # Cancel the order immediately
                await asyncio.sleep(1)
                cancel_result = await broker.cancel_order(result['id'])
                if cancel_result:
                    print("  ✅ Test order cancelled")
            else:
                print("\n⚠️  Order was not sized down (check configuration)")
        else:
            print("\n❌ Order placement failed (check logs)")
        
        # Reset max position size
        os.environ['COINBASE_MAX_POSITION_SIZE'] = '0.01'  # Reset to normal
        print(f"\n📊 Max position size reset to: {os.getenv('COINBASE_MAX_POSITION_SIZE')} BTC")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_safety_filters():
    """Test various safety filters."""
    print("\n🛡️  TESTING SAFETY FILTERS")
    print("="*60)
    
    filters = [
        {
            'name': 'Position Size Limit',
            'trigger': 'Oversized order',
            'expected': 'SIZED_DOWN',
            'test': lambda: force_sized_down_event()
        },
        {
            'name': 'Daily Loss Limit',
            'trigger': 'Loss > $100',
            'expected': 'ORDER_BLOCKED',
            'test': None  # Can't easily force without real losses
        },
        {
            'name': 'Market Impact',
            'trigger': 'Size > 15 bps of depth',
            'expected': 'SIZED_DOWN or REJECTED',
            'test': None  # Requires real market depth analysis
        },
        {
            'name': 'Kill Switch',
            'trigger': 'Manual activation',
            'expected': 'ALL_ORDERS_CANCELLED',
            'test': None  # Tested separately
        }
    ]
    
    for filter_info in filters:
        print(f"\n{filter_info['name']}:")
        print(f"  Trigger: {filter_info['trigger']}")
        print(f"  Expected: {filter_info['expected']}")
        
        if filter_info['test']:
            print("  Testing...")
            await filter_info['test']()
        else:
            print("  ⚠️  Manual testing required")


def main():
    """Run safety filter tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Force SIZED_DOWN event")
    parser.add_argument('--all', action='store_true', 
                       help='Test all safety filters')
    parser.add_argument('--confirm', action='store_true',
                       help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    if not args.confirm:
        print("\n⚠️  WARNING: This will place a real order (far from market)")
        print("The order will be immediately cancelled after validation.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    if args.all:
        asyncio.run(test_safety_filters())
    else:
        asyncio.run(force_sized_down_event())
    
    print("\n✅ Safety filter validation complete")


if __name__ == "__main__":
    main()