"""Core Coinbase data provider built from modular mixins."""

from __future__ import annotations

import os
from typing import Any, Literal, cast

from gpt_trader.data_providers import DataProvider
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from gpt_trader.config.runtime_settings import RuntimeSettings, load_runtime_settings
from gpt_trader.utilities.logging_patterns import get_logger

from .historical import CoinbaseHistoricalDataMixin
from .pricing import CoinbasePricingMixin
from .streaming import CoinbaseStreamingMixin
from .symbols import CoinbaseSymbolMixin

logger = get_logger(__name__, component="coinbase_provider")

TruthFlag = Literal["1", "true", "yes", "on"]


class CoinbaseDataProvider(
    CoinbaseStreamingMixin,
    CoinbasePricingMixin,
    CoinbaseHistoricalDataMixin,
    CoinbaseSymbolMixin,
    DataProvider,
):
    """
    Coinbase real-time data provider using the Coinbase API.

    Supports both REST and WebSocket streaming with caching and mock fallbacks.
    """

    def __init__(
        self,
        client: CoinbaseClient | None = None,
        adapter: Any | None = None,
        enable_streaming: bool = False,
        cache_ttl: int = 5,
        *,
        settings: RuntimeSettings | None = None,
    ) -> None:
        import warnings
        warnings.warn(
            "CoinbaseDataProvider is deprecated and will be removed. "
            "Please use 'gpt_trader.features.brokerages.coinbase.CoinbaseClient' "
            "or obtain the broker via 'ApplicationContainer'.",
            DeprecationWarning,
            stacklevel=2
        )
        runtime_settings = settings or load_runtime_settings()
        self._settings = runtime_settings

        sandbox_requested = runtime_settings.coinbase_sandbox_enabled
        sandbox = False
        if sandbox_requested:
            logger.warning(
                "Sandbox flag set while requesting real market data; forcing production endpoints.",
                operation="provider_init",
                status="sandbox_override",
            )

        mode_value = (runtime_settings.coinbase_api_mode or "advanced").lower()
        api_mode: Literal["advanced", "exchange"]
        if mode_value == "exchange":
            logger.warning(
                "Exchange API mode requested; defaulting to Advanced Trade endpoints for market data.",
                operation="provider_init",
                status="mode_override",
            )
            api_mode = "exchange"
        else:
            api_mode = "advanced"

        if client is None:
            # Initialize default client if not provided
            api_key = os.environ.get("COINBASE_API_KEY")
            api_secret = os.environ.get("COINBASE_API_SECRET")
            if api_key and api_secret:
                self.client = CoinbaseClient(api_key=api_key, api_secret=api_secret)
            else:
                self.client = None
        else:
            self.client = client

        if adapter is None:
            config = APIConfig(
                api_key="",
                api_secret="",
                passphrase="",
                base_url=self.client.base_url,
                ws_url="wss://advanced-trade-ws.coinbase.com",
                api_mode=cast(Literal["advanced", "exchange"], self.client.api_mode),
                sandbox=sandbox,
                auth_type="HMAC",
                enable_derivatives=False,
            )
            self.adapter = CoinbaseBrokerage(config=config, settings=runtime_settings)
        else:
            self.adapter = adapter

        streaming_env = runtime_settings.raw_env.get("COINBASE_ENABLE_STREAMING")
        env_streaming = bool(
            streaming_env
            and streaming_env.strip().lower()
            in cast(tuple[TruthFlag, ...], ("1", "true", "yes", "on"))
        )

        effective_streaming = enable_streaming or env_streaming
        self._initialize_streaming(effective_streaming, cache_ttl)
        self._initialize_historical_cache(cache_ttl)

        streaming_status = "enabled" if self.enable_streaming else "disabled"
        logger.info(
            "CoinbaseDataProvider initialised",
            operation="provider_init",
            streaming=streaming_status,
        )

    def __enter__(self) -> "CoinbaseDataProvider":
        self.start_streaming()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        self.stop_streaming()


__all__ = ["CoinbaseDataProvider"]
