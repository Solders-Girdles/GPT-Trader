"""Live integration coverage for Coinbase permissions endpoint."""

from __future__ import annotations

import pytest

from gpt_trader.features.brokerages.coinbase.auth import CDPJWTAuth
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient


@pytest.mark.integration
@pytest.mark.real_api
def test_live_permission_check(coinbase_cdp_credentials) -> None:
    auth = CDPJWTAuth(
        api_key_name=coinbase_cdp_credentials.api_key,
        private_key_pem=coinbase_cdp_credentials.private_key,
        base_host="api.coinbase.com",
    )

    client = CoinbaseClient(
        base_url="https://api.coinbase.com",
        auth=auth,
        api_mode="advanced",
    )

    perms = client.get_key_permissions()

    assert "can_view" in perms
    assert "can_trade" in perms
    assert "can_transfer" in perms
    assert "portfolio_uuid" in perms
    assert "portfolio_type" in perms
