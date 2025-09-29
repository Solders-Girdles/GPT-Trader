#!/usr/bin/env python3
"""
Canary Reduce-Only Smoke Test (Production, Advanced Trade)

Places a reduce-only limit order far from mark, verifies ACK and then cancels.
Defaults to PREVIEW mode (no live order). Use --live to actually place.

Usage:
  # Dry-run preview (safe)
  poetry run python scripts/canary_reduce_only_test.py --symbol BTC-PERP --price 10 --quantity 0.001

  # Live (use only with proper guardrails configured)
  poetry run python scripts/canary_reduce_only_test.py --live --symbol BTC-PERP --price 10 --quantity 0.001
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime


def main() -> int:
    parser = argparse.ArgumentParser(description="Reduce-Only Canary Smoke Test")
    parser.add_argument("--symbol", default="BTC-PERP", help="Perpetual symbol, e.g., BTC-PERP")
    parser.add_argument("--price", type=float, default=10.0, help="Limit price (far from market)")
    parser.add_argument("--quantity", type=float, default=0.001, help="Order quantity")
    parser.add_argument("--live", action="store_true", help="Place live order (otherwise preview)")
    args = parser.parse_args()

    from bot_v2.features.brokerages.coinbase.client import CoinbaseClient, create_cdp_jwt_auth

    # Build CDP auth from env (prod)
    import os

    api_key_name = os.getenv("COINBASE_PROD_CDP_API_KEY") or os.getenv("COINBASE_CDP_API_KEY")
    private_key_pem = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY") or os.getenv(
        "COINBASE_CDP_PRIVATE_KEY"
    )
    if not api_key_name or not private_key_pem:
        print("❌ Missing CDP credentials in environment")
        return 1

    auth = create_cdp_jwt_auth(
        api_key_name=api_key_name,
        private_key_pem=private_key_pem,
        base_url="https://api.coinbase.com",
    )
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")

    print("\n=== CANARY REDUCE-ONLY TEST ===")
    print(f"Symbol: {args.symbol}")
    print(f"Quantity: {args.quantity}")
    print(f"Price: {args.price}")
    print(f"Mode: {'LIVE' if args.live else 'PREVIEW'}")

    # Order payload (reduce-only limit, GTC)
    order = {
        "product_id": args.symbol,
        "side": "sell",  # reduce-only sell to close longs; adjust as needed
        "order_configuration": {
            "limit_limit_gtc": {
                "base_size": str(args.quantity),
                "limit_price": str(args.price),
                "post_only": True,
                "reduce_only": True,
            }
        },
    }

    try:
        t0 = datetime.now()
        if args.live:
            resp = client.place_order(order)
        else:
            resp = client.preview_order(order)
        dt_ms = (datetime.now() - t0).total_seconds() * 1000
        print(f"✅ Order {'placed' if args.live else 'previewed'} in {dt_ms:.0f}ms")
        print(str(resp)[:300] + ("..." if len(str(resp)) > 300 else ""))

        if args.live:
            order_id = resp.get("order_id") or resp.get("id")
            if not order_id:
                print("⚠️  No order_id in response; cannot cancel")
                return 1
            time.sleep(1.0)
            cancel = client.cancel_orders([order_id])
            print(f"✅ Cancel requested: {cancel}")
        return 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
