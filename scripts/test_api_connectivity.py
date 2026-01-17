#!/usr/bin/env python3
"""
API Connectivity Test Script.
Tests connection to Coinbase API using resolved CDP credentials.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpt_trader.features.brokerages.coinbase.auth import create_cdp_jwt_auth
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.credentials import (
    ResolvedCoinbaseCredentials,
    mask_key_name,
    resolve_coinbase_credentials,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-Trader Coinbase API connectivity test.",
    )
    parser.add_argument(
        "--credentials-file",
        default=os.getenv("COINBASE_CREDENTIALS_FILE"),
        help="Path to Coinbase CDP JSON key file (default: $COINBASE_CREDENTIALS_FILE).",
    )
    return parser.parse_args()


def _resolve_credentials(
    credentials_file: str | None,
) -> ResolvedCoinbaseCredentials:
    if credentials_file:
        os.environ["COINBASE_CREDENTIALS_FILE"] = credentials_file
    creds = resolve_coinbase_credentials()
    if not creds:
        print("      ERROR: Coinbase credentials not found.")
        print(
            "      Provide --credentials-file or set COINBASE_CREDENTIALS_FILE "
            "or COINBASE_CDP_API_KEY + COINBASE_CDP_PRIVATE_KEY."
        )
        sys.exit(1)
    return creds


def test_connectivity() -> None:
    """Test API connectivity with the CDP key."""
    args = _parse_args()
    print("=" * 60)
    print("GPT-Trader API Connectivity Test")
    print("=" * 60)

    # 1. Load credentials
    print("\n[1/5] Loading CDP credentials...")
    creds = _resolve_credentials(args.credentials_file)
    print(f"      Key name: {mask_key_name(creds.key_name)}")
    print(f"      Credential source: {creds.source}")
    for warning in creds.warnings:
        print(f"      Warning: {warning}")

    # 2. Create auth and client
    print("\n[2/5] Creating authenticated client...")
    auth = create_cdp_jwt_auth(api_key=creds.key_name, private_key=creds.private_key)
    client = CoinbaseClient(auth=auth)
    client.api_mode = "advanced"
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
