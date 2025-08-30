#!/usr/bin/env python3
"""
Live smoke test for CoinbaseBrokerage with CDP JWT authentication.
"""

import os
import sys
import asyncio
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from src.bot_v2.features.brokerages.coinbase.models import APIConfig
from src.bot_v2.features.core.interfaces import MarketType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_smoke_test():
    """
    Initializes CoinbaseBrokerage with CDP JWT and lists perpetual products.
    """
    logger.info("🚀 Starting Live Smoke Test...")

    cdp_api_key = os.getenv("COINBASE_CDP_API_KEY")
    cdp_private_key_path = os.getenv("COINBASE_CDP_PRIVATE_KEY_PATH")

    if not cdp_api_key or not cdp_private_key_path:
        logger.error("❌ COINBASE_CDP_API_KEY and COINBASE_CDP_PRIVATE_KEY_PATH must be set.")
        return

    try:
        with open(cdp_private_key_path, 'r') as f:
            cdp_private_key = f.read()
    except FileNotFoundError:
        logger.error(f"❌ Private key file not found at: {cdp_private_key_path}")
        return

    config = APIConfig(
        api_key="",
        api_secret="",
        passphrase="",
        base_url="",
        cdp_api_key=cdp_api_key,
        cdp_private_key=cdp_private_key,
        api_mode='advanced',
        sandbox=True,
        enable_derivatives=True
    )

    broker = CoinbaseBrokerage(config)

    logger.info("📡 Attempting to list perpetual products...")
    products = broker.list_products(market=MarketType.PERPETUAL)

    if products:
        logger.info(f"✅ Successfully fetched {len(products)} perpetual products.")
        for product in products:
            logger.info(f"  - {product.symbol}")
    else:
        logger.error("❌ Failed to fetch perpetual products.")

if __name__ == "__main__":
    asyncio.run(run_smoke_test())
