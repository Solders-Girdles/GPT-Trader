#!/usr/bin/env python3
"""
Test script for HybridPaperBroker.
Verifies real market data fetching and simulated order execution.
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpt_trader.orchestration.hybrid_paper_broker import HybridPaperBroker


def load_cdp_key(path: str | None = None) -> tuple[str, str]:
    """Load CDP API key from secrets file."""
    if path is None:
        project_root = Path(__file__).parent.parent
        path = str(project_root / "secrets" / "November2025APIKey.json")
    with open(path) as f:
        data = json.load(f)
    return data["name"], data["privateKey"]


def test_paper_broker() -> None:
    """Test the HybridPaperBroker with real market data."""
    print("=" * 60)
    print("HybridPaperBroker Test")
    print("=" * 60)

    # 1. Load credentials
    print("\n[1/6] Loading CDP credentials...")
    api_key, private_key = load_cdp_key()
    print("      Credentials loaded")

    # 2. Create paper broker
    print("\n[2/6] Creating HybridPaperBroker...")
    broker = HybridPaperBroker(
        api_key=api_key,
        private_key=private_key,
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
