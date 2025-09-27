#!/usr/bin/env python3
"""
Advanced Trade WebSocket Authentication Test (Production)

Validates JWT generation and user-channel subscription on the
advanced-trade-ws endpoint. Does not place any orders.

Usage:
  poetry run python scripts/ws_auth_test.py [--duration 30]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import ssl
import sys
from datetime import datetime, timezone


def _get_env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key, default)
    return val if val is not None and val != "" else default


async def run_test(duration: int) -> int:
    # Load CDP credentials (prod) with fallback to legacy names
    api_key_name = (
        _get_env("COINBASE_PROD_CDP_API_KEY")
        or _get_env("COINBASE_CDP_API_KEY")
    )
    private_key_pem = (
        _get_env("COINBASE_PROD_CDP_PRIVATE_KEY")
        or _get_env("COINBASE_CDP_PRIVATE_KEY")
    )
    if not api_key_name or not private_key_pem:
        print("❌ Missing CDP credentials (COINBASE_PROD_CDP_API_KEY/PRIVATE_KEY)")
        return 1

    from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2

    auth = CDPAuthV2(api_key_name=api_key_name, private_key_pem=private_key_pem)

    url = "wss://advanced-trade-ws.coinbase.com"
    ssl_ctx = ssl.create_default_context()

    try:
        import websockets
    except ImportError:
        print("websockets package missing. Install with: pip install websockets")
        return 1

    print(f"Connecting to {url} (duration={duration}s)...")
    async with websockets.connect(url, ssl=ssl_ctx, ping_interval=None, close_timeout=10) as ws:
        sub = {
            "type": "subscribe",
            "channel": "user",
            "jwt": auth.generate_jwt("GET", "/api/v3/brokerage/accounts"),
            "product_ids": ["BTC-PERP"],
        }
        await ws.send(json.dumps(sub))
        print("📤 Sent user-channel subscribe")

        # Read initial response and then keep the socket for a bit
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            print(f"First response: {msg[:160]}...")
        except asyncio.TimeoutError:
            print("⚠️  No immediate response (may be normal)")

        # Keepalive read loop
        end = asyncio.get_event_loop().time() + duration
        received = 0
        while asyncio.get_event_loop().time() < end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                received += 1
                if received <= 3:
                    print(f"▶ {msg[:160]}...")
            except asyncio.TimeoutError:
                # No messages in window; continue
                pass
        print(f"✅ Completed with {received} message(s) in {duration}s")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Advanced Trade WS Auth Test")
    parser.add_argument("--duration", type=int, default=30, help="Duration to keep the WS open (seconds)")
    args = parser.parse_args()

    print("\n=== Advanced Trade WebSocket Auth Test ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("This test is read-only and does not place orders.")
    return asyncio.run(run_test(args.duration))


if __name__ == "__main__":
    sys.exit(main())

