#!/usr/bin/env python3
"""
Canary Reduce-Only Smoke Test (Production, Advanced Trade)

Places a reduce-only limit order far from mark, verifies ACK and then cancels.
Defaults to PREVIEW mode (no live order). Live placement requires both --live
and --confirm-human-approved-live-order.

Reduce-only semantics apply to derivatives positions, so --symbol must name a
derivatives product and is required (no default): CFM contract symbols carry
expiries and roll (see CFM_SYMBOL_MAPPING in
src/gpt_trader/features/live_trade/symbols.py for the active contracts), so
pass the current contract explicitly. A spot symbol (e.g. BTC-USD) would be
rejected or give false confidence that the reduce-only path was exercised.
Retired INTX -PERP symbols are coerced to spot elsewhere in the stack and must
not be used here.

Usage:
  # Dry-run preview (safe); BTC-20DEC30-CDE is an example CFM contract —
  # substitute the currently active contract from CFM_SYMBOL_MAPPING
  uv run python scripts/monitoring/canary_reduce_only_test.py --symbol BTC-20DEC30-CDE --price 10 --quantity 0.001

  # Live (requires human-approved execution confirmation)
  uv run python scripts/monitoring/canary_reduce_only_test.py --live --confirm-human-approved-live-order --symbol BTC-20DEC30-CDE --price 10 --quantity 0.001
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

from gpt_trader.features.brokerages.coinbase.auth import create_cdp_jwt_auth
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.credentials import (
    ResolvedCoinbaseCredentials,
    mask_key_name,
    resolve_coinbase_credentials,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reduce-Only Canary Smoke Test")
    parser.add_argument(
        "--symbol",
        required=True,
        help=(
            "Derivatives product id (reduce-only requires a derivatives symbol; "
            "see CFM_SYMBOL_MAPPING for the active CFM contract, e.g. BTC-20DEC30-CDE)"
        ),
    )
    parser.add_argument("--price", type=float, default=10.0, help="Limit price (far from market)")
    parser.add_argument("--quantity", type=float, default=0.001, help="Order quantity")
    parser.add_argument("--live", action="store_true", help="Place live order (otherwise preview)")
    parser.add_argument(
        "--confirm-human-approved-live-order",
        action="store_true",
        help="Confirm that this live order test has passed the human-approved execution gate",
    )
    parser.add_argument(
        "--credentials-file",
        default=os.getenv("COINBASE_CREDENTIALS_FILE"),
        help="Path to Coinbase CDP JSON key file (default: $COINBASE_CREDENTIALS_FILE)",
    )
    return parser.parse_args(argv)


def _resolve_credentials(
    credentials_file: str | None,
) -> ResolvedCoinbaseCredentials:
    if credentials_file:
        os.environ["COINBASE_CREDENTIALS_FILE"] = credentials_file
    creds = resolve_coinbase_credentials()
    if not creds:
        print("❌ Missing CDP credentials (set COINBASE_CREDENTIALS_FILE or env vars)")
        sys.exit(1)
    return creds


def _live_gate_confirmed(args: argparse.Namespace) -> bool:
    if not args.live:
        return True
    if args.confirm_human_approved_live_order:
        return True
    print(
        "❌ Live reduce-only test requires --confirm-human-approved-live-order "
        "after the human-approved execution gate is satisfied."
    )
    return False


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not _live_gate_confirmed(args):
        return 2
    creds = _resolve_credentials(args.credentials_file)

    auth = create_cdp_jwt_auth(api_key=creds.key_name, private_key=creds.private_key)
    client = CoinbaseClient(auth=auth)
    client.api_mode = "advanced"

    print("\n=== CANARY REDUCE-ONLY TEST ===")
    print(f"Credential: {mask_key_name(creds.key_name)} ({creds.source})")
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
