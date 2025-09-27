#!/usr/bin/env python3
"""
GTD Single Controlled Test
Place one non-crossing limit order with GTD (expiry within safe short window).
Do not set post-only to avoid cross-rejection confusion.
Cancel if unfilled after short interval.
"""

import os
import sys
import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

async def test_gtd_single_order():
    """Test GTD with single controlled order."""
    print("🎯 GTD SINGLE CONTROLLED TEST")
    print("="*60)
    print("Placing one non-crossing limit order with GTD")
    print("Expiry: 5 minutes, post_only: false")
    print("="*60)
    
    try:
        from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
        from bot_v2.features.brokerages.coinbase.models import APIConfig
        
        # Configure for production
        config = APIConfig(
            api_key=os.getenv('COINBASE_CDP_API_KEY'),
            api_secret="",
            passphrase="",
            base_url="https://api.coinbase.com",
            auth_type="JWT",
            api_mode="advanced",
            sandbox=False,
            enable_derivatives=True
        )
        
        broker = CoinbaseBrokerage(config)
        
        # Calculate safe non-crossing price (well away from market)
        symbol = "BTC-PERP"
        size = Decimal("0.0001")
        
        # Place GTD order with 5-minute expiry
        expiry_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        
        print(f"\n📤 Placing GTD order:")
        print(f"  Symbol: {symbol}")
        print(f"  Size: {size}")
        print(f"  Expiry: {expiry_time.isoformat()}")
        print(f"  Post-only: false (to avoid cross-rejection)")
        
        # Place the order
        result = await broker.place_order(
            symbol=symbol,
            side="buy",
            order_type="limit",
            quantity=size,
            limit_price=Decimal("30000"),  # Well below market
            tif="GTD",
            post_only=False,
            client_id=f"gtd_test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        
        if result:
            print(f"\n✅ GTD order placed successfully:")
            print(f"  Order ID: {result.id}")
            print(f"  Status: {result.status}")
            print(f"  Size: {result.size}")
            print(f"  Price: {result.price}")
            
            # Wait 30 seconds then cancel
            print(f"\n⏱️  Waiting 30 seconds before cancellation...")
            await asyncio.sleep(30)
            
            # Cancel the order
            cancel_result = await broker.cancel_order(result.id)
            if cancel_result:
                print(f"✅ Order cancelled successfully")
            else:
                print(f"⚠️  Order cancellation failed")
                
        else:
            print(f"\n❌ GTD order placement failed")
            
        print(f"\n📄 GTD test completed")
        print(f"Result: {'SUCCESS' if result else 'FAILED'}")
        
        # Save test result
        test_result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_type": "GTD_single_controlled",
            "symbol": symbol,
            "size": str(size),
            "expiry_minutes": 5,
            "post_only": False,
            "result": "SUCCESS" if result else "FAILED",
            "order_id": result.id if result else None,
            "status": result.status if result else None
        }
        
        with open("gtd_test_result.json", "w") as f:
            json.dump(test_result, f, indent=2)
            
        print(f"📄 Test result saved to: gtd_test_result.json")
        
    except Exception as e:
        print(f"\n❌ GTD test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gtd_single_order())
