"""Shared helpers for Coinbase websocket-focused tests."""

from __future__ import annotations

from typing import Any

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig


def make_adapter(**overrides: Any) -> CoinbaseBrokerage:
    """Instantiate a brokerage adapter with sensible defaults for websocket tests."""
    config = APIConfig(
        api_key=overrides.get("api_key", "test"),
        api_secret=overrides.get("api_secret", "test"),
        passphrase=overrides.get("passphrase"),
        base_url=overrides.get("base_url", "https://api.coinbase.com"),
        sandbox=overrides.get("sandbox", False),
        api_mode=overrides.get("api_mode", "advanced"),
        enable_derivatives=overrides.get("enable_derivatives", True),
        auth_type=overrides.get("auth_type"),
        cdp_api_key=overrides.get("cdp_api_key"),
        cdp_private_key=overrides.get("cdp_private_key"),
    )
    return CoinbaseBrokerage(config)
