#!/usr/bin/env python3
"""
Stage 3: Controlled test for a single stop-limit order.

Places a stop-limit order far from the market, verifies its status,
and cancels it after a short interval.
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_stop_limit_test(cdp_key: str, cdp_key_path: str, sandbox: bool):
    """Runs the stop-limit single order test with robust JWT auth."""
    logging.info("🚀 Starting Stage 3 Stop-Limit Order Test")

    try:
        # --- 1. Configuration & Environment Assertion ---
        logging.info(f"  Mode: {'Sandbox' if sandbox else 'Production'}")
        if not cdp_key or not cdp_key_path:
            logging.error("❌ --cdp-key and --cdp-key-path are required for JWT auth.")
            return

        pem_path = Path(cdp_key_path).expanduser()
        if not pem_path.is_file():
            logging.error(f"❌ PEM file not found at: {pem_path}")
            return
        
        pem_content = pem_path.read_text()
        logging.info(f"  Successfully read PEM key from {pem_path}")

        config = APIConfig(
            cdp_api_key=cdp_key,
            cdp_private_key=pem_content,
            auth_type="JWT",
            api_mode="advanced",
            sandbox=sandbox,
            enable_derivatives=True
        )
        
        broker = CoinbaseBrokerage(config)
        symbol = "BTC-PERP"

        # --- 2. Early Auth/Connection Test ---
        logging.info(f"  Using Base URL: {broker.endpoints.base_url}")
        logging.info(f"  Using WS URL: {broker.endpoints.ws_url}")
        logging.info("  Connecting to broker to validate auth...")
        
        if not broker.connect():
            logging.error("❌ Broker connection failed. Check API keys and permissions.")
            return
        logging.info("✅ Broker connection successful.")
        
        # --- 3. Get Current Market Price ---
        logging.info(f"Fetching current market price for {symbol}...")
        quote = broker.get_quote(symbol)
        if not quote or not quote.last:
            logging.error(f"❌ Could not fetch a valid quote for {symbol}.")
            return
        
        current_price = quote.last
        logging.info(f"Current price for {symbol}: ${current_price:.2f}")

        # --- 4. Place Stop-Limit Order (far from market) ---
        side = "sell"
        quantity = Decimal("0.01")
        # Place stop 20% below current price
        stop_price = current_price * Decimal("0.8")
        limit_price = stop_price * Decimal("0.99")

        logging.info(f"Placing {side} stop-limit order for {quantity} {symbol}...")
        logging.info(f"  Stop Price:  ${stop_price:.2f}")
        logging.info(f"  Limit Price: ${limit_price:.2f}")

        order = broker.place_order(
            symbol=symbol,
            side=side,
            order_type="stop_limit",
            quantity=quantity,
            stop_price=stop_price,
            limit_price=limit_price,
            client_id=f"stage3_stop_test_{int(datetime.now(timezone.utc).timestamp())}"
        )

        if not order or not order.id:
            logging.error("❌ Failed to place stop-limit order.")
            return

        logging.info(f"✅ Order placed successfully. Order ID: {order.id}")
        order_id = order.id

        # --- 5. Verify Trigger Tracking ---
        logging.info("Verifying trigger tracking (waiting 15 seconds)...")
        await asyncio.sleep(15)

        retrieved_order = broker.get_order(order_id)
        if not retrieved_order:
            logging.error(f"❌ Could not retrieve order {order_id} to verify status.")
        else:
            logging.info(f"  Order Status: {retrieved_order.status}")
            logging.info(f"  Trigger Status: {retrieved_order.trigger_status}")
            if retrieved_order.status == "OPEN" and retrieved_order.trigger_status == "INVALID_ORDER_TYPE":
                 logging.info("✅ Trigger status correctly reflects open stop order.")
            else:
                 logging.warning("⚠️  Unexpected trigger status. Please review.")
                 logging.warning(f"  Full order details: {retrieved_order}")


        # --- 6. Cancel Order ---
        logging.info(f"Cancelling order {order_id}...")
        cancelled = broker.cancel_order(order_id)

        if not cancelled:
            logging.error("❌ Failed to cancel the order.")
        else:
            logging.info("✅ Order cancelled successfully.")

        # --- 7. Final Verification ---
        logging.info("Verifying final order status...")
        final_order_status = broker.get_order(order_id)
        if final_order_status and final_order_status.status == "CANCELLED":
            logging.info(f"✅ Final order status is CANCELLED as expected.")
        else:
            logging.error(f"❌ Final order status is not CANCELLED. Current status: {final_order_status.status if final_order_status else 'UNKNOWN'}")

        logging.info("🏁 Stop-Limit Order Test Complete.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3 Stop-Limit Order Test with JWT Auth")
    parser.add_argument("--cdp-key", type=str, required=True, help="Coinbase CDP API Key Name")
    parser.add_argument("--cdp-key-path", type=str, required=True, help="Path to the PEM private key file")
    parser.add_argument("--sandbox", action="store_true", help="Use the sandbox environment")
    args = parser.parse_args()
    asyncio.run(run_stop_limit_test(args.cdp_key, args.cdp_key_path, args.sandbox))