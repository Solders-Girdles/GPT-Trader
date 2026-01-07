"""Brokerage factory for creating broker instances.

This module provides the canonical factory function for creating brokerage
clients. It handles credential resolution, mock mode detection, and proper
dependency wiring.
"""

from __future__ import annotations

from gpt_trader.app.config import BotConfig
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.credentials import resolve_coinbase_credentials
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.orchestration.deterministic_broker import DeterministicBroker
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="brokerage")


def create_brokerage(
    event_store: EventStore,
    market_data: MarketDataService,
    product_catalog: ProductCatalog,
    config: BotConfig,
) -> tuple[CoinbaseClient | DeterministicBroker, EventStore, MarketDataService, ProductCatalog]:
    """
    Factory function to create the brokerage and verify dependencies.

    Returns DeterministicBroker when mock_broker=True.
    """
    # Check for mock mode FIRST - before credential validation
    if config.mock_broker:
        from gpt_trader.orchestration.deterministic_broker import DeterministicBroker

        return DeterministicBroker(), event_store, market_data, product_catalog

    creds = resolve_coinbase_credentials()
    if not creds:
        raise ValueError(
            "Coinbase Credentials not found. Set COINBASE_CREDENTIALS_FILE to a JSON key file, "
            "or set COINBASE_CDP_API_KEY + COINBASE_CDP_PRIVATE_KEY "
            "(legacy: COINBASE_API_KEY_NAME + COINBASE_PRIVATE_KEY)."
        )

    for warning in creds.warnings:
        logger.warning("Coinbase credential configuration: %s", warning)
    logger.info("Using Coinbase credentials from %s (%s)", creds.source, creds.masked_key_name)

    auth_client = SimpleAuth(key_name=creds.key_name, private_key=creds.private_key)

    broker = CoinbaseClient(
        auth=auth_client,
    )
    return broker, event_store, market_data, product_catalog
