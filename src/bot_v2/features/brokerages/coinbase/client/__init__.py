"""Composable Coinbase REST client and supporting mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bot_v2.config import get_config
from bot_v2.features.brokerages.coinbase.auth import (
    AuthStrategy,
    CDPJWTAuth,
    CoinbaseAuth,
    build_rest_auth,
    build_ws_auth_provider,
    create_cdp_auth,
    create_cdp_jwt_auth,
)
from bot_v2.features.brokerages.coinbase.client.accounts import AccountClientMixin
from bot_v2.features.brokerages.coinbase.client.base import CoinbaseClientBase
from bot_v2.features.brokerages.coinbase.client.market import MarketDataClientMixin
from bot_v2.features.brokerages.coinbase.client.orders import OrderClientMixin
from bot_v2.features.brokerages.coinbase.client.portfolio import PortfolioClientMixin

if TYPE_CHECKING:
    from bot_v2.orchestration.runtime_settings import RuntimeSettings
else:  # pragma: no cover - runtime type alias
    RuntimeSettings = Any  # type: ignore[misc]


class CoinbaseClient(
    CoinbaseClientBase,
    MarketDataClientMixin,
    OrderClientMixin,
    AccountClientMixin,
    PortfolioClientMixin,
):
    """Thin wrapper combining the HTTP core with specialised endpoint mixins."""

    def __init__(
        self,
        base_url: str,
        auth: AuthStrategy | None = None,
        timeout: int = 30,
        api_version: str = "2024-10-24",
        rate_limit_per_minute: int = 100,
        enable_throttle: bool = True,
        api_mode: str = "advanced",
        enable_keep_alive: bool = True,
        settings: RuntimeSettings | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            auth=auth,
            timeout=timeout,
            api_version=api_version,
            rate_limit_per_minute=rate_limit_per_minute,
            enable_throttle=enable_throttle,
            api_mode=api_mode,
            enable_keep_alive=enable_keep_alive,
            settings=settings,
        )


__all__ = [
    "AuthStrategy",
    "CDPJWTAuth",
    "CoinbaseAuth",
    "CoinbaseClient",
    "CoinbaseClientBase",
    "AccountClientMixin",
    "MarketDataClientMixin",
    "OrderClientMixin",
    "PortfolioClientMixin",
    "build_rest_auth",
    "build_ws_auth_provider",
    "create_cdp_auth",
    "create_cdp_jwt_auth",
    "get_config",
]
