"""Brokerage factory for creating broker instances.

This module provides the canonical factory function for creating brokerage
clients. It handles credential resolution, mock mode detection, and proper
dependency wiring.
"""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.app.config import BotConfig
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.credentials import resolve_coinbase_credentials
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.features.brokerages.core.guarded_broker import GuardedBroker
from gpt_trader.features.brokerages.mock import DeterministicBroker
from gpt_trader.features.brokerages.paper import HybridPaperBroker
from gpt_trader.features.brokerages.read_only import ReadOnlyBroker
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="brokerage")


def create_brokerage(
    event_store: EventStore,
    market_data: MarketDataService,
    product_catalog: ProductCatalog,
    config: BotConfig,
) -> tuple[
    CoinbaseClient | DeterministicBroker | ReadOnlyBroker | HybridPaperBroker,
    EventStore,
    MarketDataService,
    ProductCatalog,
]:
    """
    Factory function to create the brokerage and verify dependencies.

    Returns DeterministicBroker when mock_broker=True. In dry-run mode, wraps the
    broker with ReadOnlyBroker to suppress write calls.
    """
    broker: CoinbaseClient | DeterministicBroker | ReadOnlyBroker | HybridPaperBroker

    if config.paper_fills and config.mock_broker:
        raise ValueError("Cannot use paper_fills with mock_broker; pick one")
    if config.paper_fills and config.dry_run:
        raise ValueError("Cannot use paper_fills with dry_run; pick one simulation mode")

    # Check for mock mode FIRST - before credential validation
    if config.mock_broker:
        broker = DeterministicBroker()
    elif config.paper_fills:
        creds = resolve_coinbase_credentials()
        if not creds:
            raise ValueError(
                "paper_fills mode requires Coinbase credentials for market data. "
                "Set COINBASE_CREDENTIALS_FILE to a JSON key file, or set "
                "COINBASE_CDP_API_KEY + COINBASE_CDP_PRIVATE_KEY."
            )

        for warning in creds.warnings:
            logger.warning("Coinbase credential configuration: %s", warning)
        logger.info(
            "Using Coinbase credentials from %s (%s) for paper fills",
            creds.source,
            creds.masked_key_name,
        )

        auth_client = SimpleAuth(key_name=creds.key_name, private_key=creds.private_key)
        market_client = CoinbaseClient(auth=auth_client)

        raw_equity = getattr(getattr(config, "risk", None), "dry_run_equity_usd", None)
        try:
            initial_equity = (
                Decimal(str(raw_equity)) if raw_equity is not None else Decimal("10000")
            )
        except Exception:
            initial_equity = Decimal("10000")

        broker = HybridPaperBroker(
            client=market_client,
            initial_equity=initial_equity,
            slippage_bps=5,
            commission_bps=Decimal("5"),
        )
        logger.info(
            "Paper fills enabled: real market data + simulated execution",
            initial_equity=float(initial_equity),
        )
    else:
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

    if config.dry_run:
        bot_id = str(config.profile) if config.profile is not None else None
        broker = ReadOnlyBroker(
            broker=broker,
            event_store=event_store,
            bot_id=bot_id,
        )
        logger.info("Dry-run enabled: broker write calls are suppressed")
    elif not config.mock_broker and not config.paper_fills:
        broker = GuardedBroker(broker, strict=True)
        logger.info("Order guard enabled for live broker execution")

    return broker, event_store, market_data, product_catalog
