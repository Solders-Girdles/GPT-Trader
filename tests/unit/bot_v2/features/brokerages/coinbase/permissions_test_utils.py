"""Shared helpers for Coinbase API permission tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.auth import CDPJWTAuth
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.models import APIConfig


def make_client(**overrides: Any) -> CoinbaseClient:
    """Construct a CoinbaseClient with optional overrides for tests."""
    auth = overrides.pop("auth", None)
    if auth is None:
        auth = Mock(spec=CDPJWTAuth)
    return CoinbaseClient(
        base_url=overrides.pop("base_url", "https://api.coinbase.com"),
        auth=auth,
        api_mode=overrides.pop("api_mode", "advanced"),
        api_version=overrides.pop("api_version", None),
    )


def make_broker(**overrides: Any) -> CoinbaseBrokerage:
    """Create a CoinbaseBrokerage with a configurable APIConfig."""
    config = APIConfig(
        api_key=overrides.pop("api_key", "test_key"),
        api_secret=overrides.pop("api_secret", "test_secret"),
        passphrase=overrides.pop("passphrase", "test_pass"),
        base_url=overrides.pop("base_url", "https://api.coinbase.com"),
        api_mode=overrides.pop("api_mode", "advanced"),
        sandbox=overrides.pop("sandbox", False),
        enable_derivatives=overrides.pop("enable_derivatives", True),
        cdp_api_key=overrides.pop("cdp_api_key", "test_cdp_key"),
        cdp_private_key=overrides.pop("cdp_private_key", "test_cdp_secret"),
        auth_type=overrides.pop("auth_type", "JWT"),
    )
    return CoinbaseBrokerage(config)
