"""Consolidated Coinbase authentication tests."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.auth import CDPJWTAuth, SimpleAuth
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from tests.unit.gpt_trader.features.brokerages.coinbase.helpers import CoinbaseBrokerage


def test_cdp_jwt_auth_generates_valid_headers(monkeypatch):
    """Test that CDPJWTAuth generates proper JWT headers."""
    # Mock JWT generation to avoid needing a real EC key
    monkeypatch.setattr(CDPJWTAuth, "generate_jwt", lambda self, method, path: "mock_jwt_token")

    auth = CDPJWTAuth(api_key="test_key", private_key="mock_private_key")
    headers = auth.get_headers("GET", "/api/v3/brokerage/products")

    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer mock_jwt_token"
    assert headers["Content-Type"] == "application/json"


def test_simple_auth_generates_valid_headers(monkeypatch):
    """Test that SimpleAuth generates proper JWT headers."""
    # Mock JWT generation to avoid needing a real EC key
    monkeypatch.setattr(SimpleAuth, "generate_jwt", lambda self, method, path: "mock_jwt_token")

    auth = SimpleAuth(key_name="test_key", private_key="mock_private_key")
    headers = auth.get_headers("POST", "/api/v3/brokerage/orders")

    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer mock_jwt_token"
    assert headers["Content-Type"] == "application/json"


def test_broker_auth_selection_uses_cdp_jwt():
    """Test that CoinbaseBrokerage uses CDPJWTAuth when CDP keys are provided."""
    config = APIConfig(
        base_url="https://api.coinbase.com",
        sandbox=True,
        api_mode="advanced",
        enable_derivatives=True,
        api_key="",
        api_secret="",
        passphrase=None,
        cdp_api_key="test_cdp_key",
        cdp_private_key="mock_private_key",
    )
    broker = CoinbaseBrokerage(config)
    assert isinstance(broker.client.auth, CDPJWTAuth)


def test_broker_auth_fallback_uses_simple_auth():
    """Test that CoinbaseBrokerage uses SimpleAuth when only api_key/api_secret provided."""
    config = APIConfig(
        base_url="https://api.coinbase.com",
        sandbox=True,
        api_mode="advanced",
        enable_derivatives=True,
        api_key="test_api_key",
        api_secret="mock_private_key",
        passphrase=None,
    )
    broker = CoinbaseBrokerage(config)
    assert isinstance(broker.client.auth, SimpleAuth)
