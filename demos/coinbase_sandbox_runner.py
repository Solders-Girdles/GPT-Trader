"""
Coinbase Sandbox Runner (opt-in; uses real network if creds are set).

Requirements:
- Set env: BROKER=coinbase, COINBASE_SANDBOX=1, COINBASE_API_KEY/SECRET[/PASSPHRASE]

Usage:
  python demos/coinbase_sandbox_runner.py

Notes:
- This runner performs a read-only smoke by default.
- To place a test limit order (and cancel), set COINBASE_RUN_ORDER_TESTS=1.
"""

import os
import time
from decimal import Decimal

from bot_v2.orchestration.broker_factory import create_brokerage
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce


def main():
    if os.getenv("BROKER", "").lower() != "coinbase":
        print("Set BROKER=coinbase to run this demo.")
        return
    if os.getenv("COINBASE_SANDBOX") != "1":
        print("Set COINBASE_SANDBOX=1 and provide sandbox API credentials.")
        return

    broker = create_brokerage()
    print("Connecting to Coinbase sandbox...")
    if not broker.connect():
        print("Failed to connect.")
        return
    print("Connected.")

    print("Listing products...")
    prods = broker.list_products()
    print(f"Products: {len(prods)}; First: {prods[0].symbol if prods else 'n/a'}")

    print("Listing balances...")
    bals = broker.list_balances()
    print([f"{b.asset}:{b.available}" for b in bals][:5])

    if os.getenv("COINBASE_RUN_ORDER_TESTS") == "1":
        symbol = os.getenv("COINBASE_ORDER_SYMBOL", "BTC-USD")
        price = Decimal(os.getenv("COINBASE_TEST_LIMIT_PRICE", "10"))
        qty = Decimal(os.getenv("COINBASE_TEST_QTY", "0.001"))
        print(f"Placing test limit order {symbol} qty={qty} price={price}...")
        o = broker.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            qty=qty,
            price=price,
            tif=TimeInForce.GTC,
            client_id=f"gpt-demo-{int(time.time())}",
        )
        print("Order:", o)
        print("Cancelling...")
        print("Cancelled:", broker.cancel_order(o.id))
    else:
        print("Skipping order tests; set COINBASE_RUN_ORDER_TESTS=1 to enable.")


if __name__ == "__main__":
    main()

