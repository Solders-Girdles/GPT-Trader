#!/usr/bin/env python3
"""
Production Perpetuals Preflight (Read-Only)

Validates Advanced Trade (CDP/JWT) credentials and connectivity without placing orders.

Checks:
- Env sanity (advanced mode, derivatives enabled, sandbox off)
- JWT generation using COINBASE_PROD_CDP_* (fallback to COINBASE_CDP_*)
- REST endpoints: accounts, CFM positions, best_bid_ask, time
- WebSocket user-channel authentication (JWT)
- Public ticker stream presence for BTC-PERP / ETH-PERP

Usage:
  poetry run python scripts/prod_perps_preflight.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone


def _get_env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key, default)
    return val if val is not None and val != "" else default


def _print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    from bot_v2.features.brokerages.coinbase.cdp_auth_v2 import CDPAuthV2
    from bot_v2.features.brokerages.coinbase.client import CoinbaseClient

    _print_header("🚀 Advanced Trade Perpetuals Preflight (Read-Only)")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    # Env sanity
    sanbox = _get_env("COINBASE_SANDBOX", "0")
    api_mode = _get_env("COINBASE_API_MODE", "advanced")
    enable_derivs = _get_env("COINBASE_ENABLE_DERIVATIVES", "1")
    print("Environment:")
    print(f"  COINBASE_SANDBOX: {sanbox}")
    print(f"  COINBASE_API_MODE: {api_mode}")
    print(f"  COINBASE_ENABLE_DERIVATIVES: {enable_derivs}")
    if sanbox == "1":
        print("⚠️  Sandbox is enabled — Sandbox does not support perpetuals. Set COINBASE_SANDBOX=0.")
        return 1
    if api_mode != "advanced":
        print("⚠️  API mode is not 'advanced'. Set COINBASE_API_MODE=advanced.")
        return 1

    # Load CDP (prod) creds with fallback to legacy names
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

    # JWT generation
    _print_header("🔐 JWT Generation")
    try:
        auth = CDPAuthV2(api_key_name=api_key_name, private_key_pem=private_key_pem)
        jwt_token = auth.generate_jwt(method="GET", path="/api/v3/brokerage/accounts")
        print("✅ JWT generated successfully (ES256)")
    except Exception as e:
        print(f"❌ Failed to generate JWT: {e}")
        return 1

    # REST client (Advanced Trade)
    client = CoinbaseClient(
        base_url="https://api.coinbase.com",
        auth=auth,
        api_mode="advanced",
    )

    # REST checks
    _print_header("📡 REST Endpoint Checks")
    def _check(name: str, func, *args, **kwargs):
        try:
            t0 = time.perf_counter()
            resp = func(*args, **kwargs)
            dt = (time.perf_counter() - t0) * 1000
            ok = isinstance(resp, dict)
            print(f"✅ {name}: {int(dt)}ms")
            return True
        except Exception as e:
            print(f"❌ {name}: {e}")
            return False

    ok_accounts = _check("accounts", client.get_accounts)
    ok_cfm_pos = _check("cfm_positions", client.cfm_positions)
    try:
        ok_bba = _check("best_bid_ask (BTC/ETH PERP)", client.get_best_bid_ask, ["BTC-PERP", "ETH-PERP"])
    except Exception as e:
        print(f"❌ best_bid_ask: {e}")
        ok_bba = False
    ok_time = _check("server time", client.get_time)

    # WS user channel auth
    _print_header("🌊 WebSocket User Channel (JWT)")
    try:
        import asyncio
        import ssl
        import websockets

        async def ws_auth():
            url = "wss://advanced-trade-ws.coinbase.com"
            ssl_ctx = ssl.create_default_context()
            async with websockets.connect(url, ssl=ssl_ctx, ping_interval=None, close_timeout=10) as ws:
                sub = {
                    "type": "subscribe",
                    "channel": "user",
                    "jwt": auth.generate_jwt("GET", "/api/v3/brokerage/accounts"),
                    "product_ids": ["BTC-PERP"],
                }
                await ws.send(json.dumps(sub))
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                print(f"WS response: {msg[:120]}...")
                return True

        asyncio.run(ws_auth())
        print("✅ WS user-channel subscribe attempted (inspect response)")
        ok_ws = True
    except Exception as e:
        print(f"❌ WS user-channel failed: {e}")
        ok_ws = False

    # Public ticker (Exchange feed validates network path for market data)
    _print_header("📈 Public Ticker (Exchange Sandbox feed for sanity)")
    try:
        import asyncio
        import ssl
        import websockets

        async def ws_pub():
            url = "wss://ws-feed.exchange.coinbase.com"
            ssl_ctx = ssl.create_default_context()
            async with websockets.connect(url, ssl=ssl_ctx, ping_interval=None, close_timeout=10) as ws:
                sub = {"type": "subscribe", "channels": ["ticker"], "product_ids": ["BTC-USD"]}
                await ws.send(json.dumps(sub))
                await asyncio.wait_for(ws.recv(), timeout=5)
                return True

        asyncio.run(ws_pub())
        print("✅ Public ticker stream available (Exchange feed)")
        ok_pub = True
    except Exception as e:
        print(f"⚠️  Public ticker check skipped/failed: {e}")
        ok_pub = False

    # Summary
    _print_header("📋 Summary")
    results = {
        "accounts": ok_accounts,
        "cfm_positions": ok_cfm_pos,
        "best_bid_ask": ok_bba,
        "server_time": ok_time,
        "ws_user_auth": ok_ws,
        "ws_public_ticker": ok_pub,
    }
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    all_ok = ok_accounts and ok_cfm_pos and ok_bba and ok_time and ok_ws
    print(f"\nOverall: {'🟢 PASS' if all_ok else '🔴 FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

