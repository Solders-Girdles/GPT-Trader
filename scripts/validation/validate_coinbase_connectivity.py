#!/usr/bin/env python3
"""
Simple connectivity validator for Coinbase Perpetuals setup.

Scenarios:
- Paper/Mock mode (PERPS_PAPER=1 or PERPS_FORCE_MOCK=1): exercises MockBroker.
- Production (Advanced Trade + JWT): connects via CoinbaseBrokerage and lists products.

Usage:
  python scripts/validation/validate_coinbase_connectivity.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def _is_true(v: Optional[str]) -> bool:
    return str(v or '').lower() in ('1', 'true', 'yes', 'on')


def main() -> int:
    os.environ.setdefault('BROKER', 'coinbase')

    paper = _is_true(os.getenv('PERPS_PAPER')) or _is_true(os.getenv('PERPS_FORCE_MOCK'))

    if paper:
        print("== Paper/Mock Mode Connectivity Check ==")
        from bot_v2.orchestration.mock_broker import MockBroker
        broker = MockBroker()
        ok = broker.connect()
        print(f"Mock connect: {ok}")
        products = broker.list_products()
        print(f"Products: {len(products)} (e.g., {[p.symbol for p in products[:3]]})")
        q = broker.get_quote('BTC-PERP')
        print(f"BTC-PERP quote: last={q.last} bid={q.bid} ask={q.ask}")
        return 0

    print("== Production Connectivity Check (Advanced Trade + JWT) ==")
    # Enforce JWT presence early for clarity
    cdp_key = os.getenv('COINBASE_PROD_CDP_API_KEY') or os.getenv('COINBASE_CDP_API_KEY')
    cdp_priv = os.getenv('COINBASE_PROD_CDP_PRIVATE_KEY') or os.getenv('COINBASE_CDP_PRIVATE_KEY')
    if not (cdp_key and cdp_priv):
        print("ERROR: Missing CDP JWT credentials. Set COINBASE_PROD_CDP_API_KEY and COINBASE_PROD_CDP_PRIVATE_KEY.")
        print("Tip: set PERPS_PAPER=1 to validate mock mode without credentials.")
        return 2

    # Require derivatives flag on
    if os.getenv('COINBASE_ENABLE_DERIVATIVES', '0') != '1':
        print("ERROR: COINBASE_ENABLE_DERIVATIVES=1 required for perps.")
        return 3

    # Force Advanced Trade and production settings for clarity
    os.environ['COINBASE_API_MODE'] = 'advanced'
    os.environ['COINBASE_SANDBOX'] = '0'

    try:
        from bot_v2.orchestration.broker_factory import create_brokerage
        broker = create_brokerage()
        ok = broker.connect()
        print(f"Connected: {ok}")
        prods = broker.list_products()
        perps = [p for p in prods if getattr(p, 'market_type', None) and p.market_type.name == 'PERPETUAL']
        print(f"Products: {len(prods)}, Perpetuals: {len(perps)}")
        if perps:
            q = broker.get_quote(perps[0].symbol)
            print(f"{perps[0].symbol} quote: last={q.last}")
        return 0
    except Exception as e:
        print(f"ERROR: Connectivity failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

