#!/usr/bin/env python3
"""
Test script for HybridPaperBroker.
Verifies real market data fetching and simulated order execution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
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
from gpt_trader.features.brokerages.paper import HybridPaperBroker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HybridPaperBroker connectivity test.",
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


def test_paper_broker() -> None:
    """Test the HybridPaperBroker with real market data."""
    args = _parse_args()
    print("=" * 60)
    print("HybridPaperBroker Test")
    print("=" * 60)

    # 1. Load credentials
    print("\n[1/6] Loading CDP credentials...")
    creds = _resolve_credentials(args.credentials_file)
    print(f"      Key name: {mask_key_name(creds.key_name)}")
    print(f"      Credential source: {creds.source}")
    for warning in creds.warnings:
        print(f"      Warning: {warning}")

    # 2. Create paper broker
    print("\n[2/6] Creating HybridPaperBroker...")
    auth = create_cdp_jwt_auth(api_key=creds.key_name, private_key=creds.private_key)
    client = CoinbaseClient(auth=auth)
    client.api_mode = "advanced"
    broker = HybridPaperBroker(
        client=client,
        initial_equity=Decimal("10000"),
        slippage_bps=5,
        commission_bps=Decimal("5"),
    )
    print(f"      Initial equity: ${broker.get_equity()}")

    # 3. Test market data
    print("\n[3/6] Testing market data (BTC-USD quote)...")
    quote = broker.get_quote("BTC-USD")
    if quote:
        print(f"      BTC-USD: bid=${quote.bid:.2f}, ask=${quote.ask:.2f}, last=${quote.last:.2f}")
    else:
        print("      ERROR: Could not fetch quote")
        sys.exit(1)

    # 4. Test simulated buy order
    print("\n[4/6] Simulating BUY order...")
    order = broker.place_order(
        symbol_or_payload="BTC-USD",
        side="buy",
        order_type="market",
        quantity=Decimal("0.01"),
    )
    print(f"      Order ID: {order.id}")
    print(f"      Status: {order.status.value}")
    print(f"      Filled: {order.filled_quantity} @ ${order.avg_fill_price:.2f}")

    # 5. Check position
    print("\n[5/6] Checking position...")
    positions = broker.list_positions()
    if positions:
        pos = positions[0]
        print(f"      Position: {pos.quantity} {pos.symbol} @ ${pos.entry_price:.2f}")
    else:
        print("      No positions (unexpected)")

    # 6. Check balance and equity
    print("\n[6/6] Checking balance and equity...")
    balances = broker.list_balances()
    for bal in balances:
        print(f"      {bal.asset}: ${bal.total:.2f}")
    print(f"      Total Equity: ${broker.get_equity():.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("PAPER BROKER TEST PASSED")
    print("=" * 60)
    status = broker.get_status()
    print(f"\nStatus: {json.dumps(status, indent=2)}")


if __name__ == "__main__":
    test_paper_broker()
