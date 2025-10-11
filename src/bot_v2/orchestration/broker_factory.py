"""
Broker factory to instantiate brokerage adapters based on config/env.
"""

from __future__ import annotations

import logging
import os
from typing import Literal, cast

from bot_v2.config import get_config
from bot_v2.features.brokerages.coinbase import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
from bot_v2.features.brokerages.core.interfaces import IBrokerage
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.persistence.event_store import EventStore

logger = logging.getLogger(__name__)


def _env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def create_brokerage(
    registry: ServiceRegistry | None = None,
    *,
    event_store: EventStore | None = None,
    market_data: MarketDataService | None = None,
    product_catalog: ProductCatalog | None = None,
) -> tuple[IBrokerage, EventStore, MarketDataService, ProductCatalog]:
    """Create a brokerage adapter based on configuration.

    Uses env var `BROKER` to select: e.g., `coinbase`.
    Coinbase credentials (env-aware, with sandbox/prod fallbacks):
      - COINBASE_API_KEY / COINBASE_PROD_API_KEY / COINBASE_SANDBOX_API_KEY
      - COINBASE_API_SECRET / COINBASE_PROD_API_SECRET / COINBASE_SANDBOX_API_SECRET
      - COINBASE_API_PASSPHRASE / COINBASE_PROD_API_PASSPHRASE / COINBASE_SANDBOX_API_PASSPHRASE
      - COINBASE_CDP_API_KEY (or COINBASE_PROD_CDP_API_KEY)
      - COINBASE_CDP_PRIVATE_KEY (or COINBASE_PROD_CDP_PRIVATE_KEY)
      - COINBASE_AUTH_TYPE ("HMAC" or "JWT")
      - COINBASE_API_BASE (defaults to Coinbase Advanced Trade v3 base or sandbox)
      - COINBASE_SANDBOX ("1" enables sandbox base URL)
      - COINBASE_API_MODE ("advanced" or "exchange" - auto-detected if not set)
    """
    broker_value = _env("BROKER") or get_config("system").get("broker") or "coinbase"
    broker = broker_value.lower()

    if broker == "coinbase":
        sandbox = _env("COINBASE_SANDBOX", "0") == "1"

        # Determine API mode - sandbox ALWAYS uses exchange mode
        if sandbox:
            api_mode: str = "exchange"
        else:
            api_mode = _env("COINBASE_API_MODE") or "advanced"

        # If not explicitly set, determine based on sandbox and base URL
        if not api_mode:
            if sandbox:
                # Sandbox defaults to exchange mode (legacy) since AT doesn't have public sandbox
                api_mode = "exchange"
                logger.warning(
                    "Sandbox mode enabled: Using legacy Exchange API. "
                    "Only basic endpoints available. For full Advanced Trade features, "
                    "use production with test account."
                )
            else:
                # Production defaults to advanced mode
                api_mode = "advanced"

        # Set base URL based on mode and sandbox
        base_url = _env("COINBASE_API_BASE")
        if not base_url:
            if api_mode == "exchange":
                base_url = (
                    "https://api-public.sandbox.exchange.coinbase.com"
                    if sandbox
                    else "https://api.exchange.coinbase.com"
                )
            else:  # advanced
                if sandbox:
                    logger.warning(
                        "Advanced Trade API does not have a public sandbox. "
                        "Consider using 'exchange' mode for sandbox testing."
                    )
                base_url = "https://api.coinbase.com"

        # Set WebSocket URL based on mode
        ws_url = _env("COINBASE_WS_URL")
        if not ws_url:
            if api_mode == "exchange":
                ws_url = (
                    "wss://ws-feed-public.sandbox.exchange.coinbase.com"
                    if sandbox
                    else "wss://ws-feed.exchange.coinbase.com"
                )
            else:  # advanced
                ws_url = "wss://advanced-trade-ws.coinbase.com"

        # Determine auth type based on API mode
        cdp_api_key = _env("COINBASE_CDP_API_KEY")
        cdp_private_key = _env("COINBASE_CDP_PRIVATE_KEY")

        if api_mode == "exchange":
            auth_type: str = "HMAC"
        elif cdp_api_key and cdp_private_key:
            auth_type = "JWT"
        else:
            auth_type = _env("COINBASE_AUTH_TYPE") or "HMAC"

        # Validate auth requirements for exchange mode
        if api_mode == "exchange" and not _env("COINBASE_API_PASSPHRASE"):
            logger.warning(
                "Exchange API mode requires passphrase for HMAC auth. "
                "Set COINBASE_API_PASSPHRASE environment variable."
            )

        # Choose credential set based on environment
        if sandbox:
            api_key = _env("COINBASE_SANDBOX_API_KEY") or _env("COINBASE_API_KEY", "")
            api_secret = _env("COINBASE_SANDBOX_API_SECRET") or _env("COINBASE_API_SECRET", "")
            passphrase = _env("COINBASE_SANDBOX_API_PASSPHRASE") or _env("COINBASE_API_PASSPHRASE")
            cdp_api_key = _env(
                "COINBASE_CDP_API_KEY"
            )  # Sandbox does not support AT; keep for completeness
            cdp_private_key = _env("COINBASE_CDP_PRIVATE_KEY")
        else:
            api_key = _env("COINBASE_PROD_API_KEY") or _env("COINBASE_API_KEY") or ""
            api_secret = _env("COINBASE_PROD_API_SECRET") or _env("COINBASE_API_SECRET") or ""
            passphrase = _env("COINBASE_PROD_API_PASSPHRASE") or _env("COINBASE_API_PASSPHRASE")
            cdp_api_key = _env("COINBASE_PROD_CDP_API_KEY") or _env("COINBASE_CDP_API_KEY")
            cdp_private_key = _env("COINBASE_PROD_CDP_PRIVATE_KEY") or _env(
                "COINBASE_CDP_PRIVATE_KEY"
            )

        api_config = APIConfig(
            api_key=api_key or "",
            api_secret=api_secret or "",
            passphrase=passphrase,
            base_url=base_url,
            sandbox=sandbox,
            ws_url=ws_url,
            enable_derivatives=_env("COINBASE_ENABLE_DERIVATIVES", "0") == "1",
            cdp_api_key=cdp_api_key,
            cdp_private_key=cdp_private_key,
            auth_type=auth_type,
            api_mode=cast(Literal["advanced", "exchange"], api_mode),
        )

        logger.info(
            "Creating Coinbase brokerage: mode=%s, sandbox=%s, auth=%s",
            api_mode,
            sandbox,
            auth_type,
        )

        broker_event_store = (
            event_store or (registry.event_store if registry else None) or EventStore()
        )
        broker_market_data = (
            market_data
            or (registry.market_data_service if registry else None)
            or MarketDataService()
        )
        broker_product_catalog = (
            product_catalog
            or (registry.product_catalog if registry else None)
            or ProductCatalog(ttl_seconds=900)
        )

        brokerage = CoinbaseBrokerage(
            api_config,
            event_store=broker_event_store,
            market_data=broker_market_data,
            product_catalog=broker_product_catalog,
        )
        return brokerage, broker_event_store, broker_market_data, broker_product_catalog

    raise ValueError(f"Unsupported broker: {broker}")
