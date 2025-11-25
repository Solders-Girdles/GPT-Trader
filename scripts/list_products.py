import os
import sys
import json
from pathlib import Path
from pprint import pprint

from gpt_trader.app.container import create_application_container
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.config.runtime_settings import RuntimeSettings, load_runtime_settings

# Placeholder config for bootstrapping
config = BotConfig(
    symbols=[], dry_run=True  # No symbols needed for listing products  # No actual trading
)

# Manually load runtime settings to get env vars
settings = load_runtime_settings()

container = create_application_container(config, settings=settings)

try:
    # Access the broker client
    broker_client = container.broker

    print("Attempting to list perpetual futures products...")
    # Call list_products with appropriate filters for perpetual futures
    # (assuming 'future' product_type and 'perpetual' contract_expiry_type)
    # Check MarketDataClientMixin for correct parameters.
    products = broker_client.list_products()

    if products:
        print(f"Found {len(products)} total products:")
        for product in products:
            print(
                f"- ID: {product.get('product_id')}, Status: {product.get('status')}, Type: {product.get('product_type')}, Quote: {product.get('quote_currency_id')}"
            )
    else:
        print("No products found at all. Check API key permissions and network.")


except Exception as e:
    print(f"FAILED to list products: {e}")
    import traceback

    traceback.print_exc()
