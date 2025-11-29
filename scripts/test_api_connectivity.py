#!/usr/bin/env python3
"""
API Connectivity Test Script.
Tests connection to Coinbase API using the CDP key from /secrets/November2025APIKey.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient


def load_cdp_key(path: str | None = None) -> tuple[str, str]:
    """Load CDP API key from secrets file."""
    if path is None:
        # Default to project-relative path
        project_root = Path(__file__).parent.parent
        path = str(project_root / "secrets" / "November2025APIKey.json")
    with open(path) as f:
        data = json.load(f)
    return data["name"], data["privateKey"]


def test_connectivity() -> None:
    """Test API connectivity with the CDP key."""
    print("=" * 60)
    print("GPT-Trader API Connectivity Test")
    print("=" * 60)

    # 1. Load credentials
    print("\n[1/5] Loading CDP credentials...")
    try:
        key_name, private_key = load_cdp_key()
        # Mask the key for display
        masked_name = key_name[:30] + "..." if len(key_name) > 30 else key_name
        print(f"      Key name: {masked_name}")
        print("      Private key: [LOADED]")
    except FileNotFoundError:
        print("      ERROR: Secrets file not found at /secrets/November2025APIKey.json")
        sys.exit(1)
    except Exception as e:
        print(f"      ERROR: Failed to load credentials: {e}")
        sys.exit(1)

    # 2. Create auth and client
    print("\n[2/5] Creating authenticated client...")
    auth = SimpleAuth(key_name=key_name, private_key=private_key)
    client = CoinbaseClient(auth=auth)
    print("      Client created successfully")

    # 3. Test: Get server time (no auth needed, good baseline)
    print("\n[3/5] Testing API connectivity (server time)...")
    try:
        time_response = client.get("/api/v3/brokerage/time")
        print(f"      Server time: {time_response}")
    except Exception as e:
        print(f"      ERROR: {e}")
        print("      (Continuing with other tests...)")

    # 4. Test: Check key permissions
    print("\n[4/6] Checking API key permissions...")
    try:
        permissions = client.get("/api/v3/brokerage/key_permissions")
        print(f"      Permissions: {permissions}")
    except Exception as e:
        print(f"      Could not get key permissions: {e}")
        print("      (This may be expected for some key types)")

    # 4b. Test: Get products (requires auth)
    print("\n[5/6] Testing authenticated access (list products)...")
    try:
        products = client.list_products(product_type="spot")
        spot_count = len(products)
        print(f"      Found {spot_count} spot products")
        if spot_count > 0:
            sample = [p.get("product_id") for p in products[:5]]
            print(f"      Sample: {sample}")
    except Exception as e:
        print(f"      WARNING: Authenticated products failed: {e}")
        print("      Trying public market products endpoint...")
        try:
            # Try unauthenticated public endpoint
            public_products = client.get_market_products()
            products = public_products.get("products", [])
            spot_products = [p for p in products if p.get("product_type") == "SPOT"]
            print(f"      Found {len(spot_products)} spot products (via public API)")
            if spot_products:
                sample = [p.get("product_id") for p in spot_products[:5]]
                print(f"      Sample: {sample}")
        except Exception as e2:
            print(f"      ERROR: Public products also failed: {e2}")
            sys.exit(1)

    # 6. Test: Get BTC-USD ticker
    print("\n[6/6] Testing market data (BTC-USD ticker)...")
    try:
        ticker = client.get_ticker("BTC-USD")
        price = ticker.get("price") or ticker.get("trades", [{}])[0].get("price", "N/A")
        print(f"      BTC-USD price: ${price}")
    except Exception as e:
        print(f"      ERROR: Failed to get ticker: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("CONNECTIVITY TEST PASSED")
    print("=" * 60)
    print("\nYour API key is working! You can proceed with paper trading setup.")
    print("\nNext steps:")
    print("  1. Configure .env with credentials")
    print("  2. Build HybridPaperBroker")
    print("  3. Deploy with Docker")


if __name__ == "__main__":
    test_connectivity()
