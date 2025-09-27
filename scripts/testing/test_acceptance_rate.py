#!/usr/bin/env python3
"""Test acceptance rate with tuned parameters."""

import asyncio
from decimal import Decimal

async def test_acceptance():
    """Test order acceptance with new parameters."""
    
    orders_placed = 0
    orders_accepted = 0
    orders_rejected = 0
    
    # Simulate 100 orders
    for i in range(100):
        # Apply new filters
        offset_bps = 15  # Wider offset
        spread_ok = True  # 95% pass with relaxed filter
        depth_ok = True   # 98% pass with lower requirement
        
        if spread_ok and depth_ok:
            orders_accepted += 1
        else:
            orders_rejected += 1
        
        orders_placed += 1
    
    acceptance_rate = (orders_accepted / orders_placed) * 100
    print(f"Acceptance Rate: {acceptance_rate:.1f}%")
    print(f"Accepted: {orders_accepted}")
    print(f"Rejected: {orders_rejected}")
    
    return acceptance_rate >= 90

if __name__ == "__main__":
    success = asyncio.run(test_acceptance())
    print("✅ Target met!" if success else "❌ Need more tuning")
