"""Shared fixtures for preflight tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.preflight.core import PreflightCheck


@pytest.fixture
def mock_healthy_client():
    """Create a mock client with healthy defaults."""
    client = MagicMock()
    client.get_resilience_status.return_value = None
    client.get_accounts.return_value = {"accounts": [{"id": "acc1"}]}
    client.list_products.return_value = [{"product_id": "BTC-USD"}]
    client.get_product.return_value = {
        "base_min_size": "0.001",
        "base_increment": "0.001",
        "quote_increment": "0.01",
    }
    client.get_ticker.return_value = {"price": "50000.00"}
    return client


@pytest.fixture
def checker():
    """Create a production checker."""
    return PreflightCheck(profile="prod")


@pytest.fixture
def force_remote_env(monkeypatch: pytest.MonkeyPatch):
    """Set environment for forced remote checks."""
    monkeypatch.setenv("COINBASE_PREFLIGHT_FORCE_REMOTE", "1")
    monkeypatch.setenv("TRADING_SYMBOLS", "BTC-USD")
    return monkeypatch


@pytest.fixture
def warn_only_env(force_remote_env: pytest.MonkeyPatch):
    """Set environment for warn-only mode with forced remote checks."""
    force_remote_env.setenv("GPT_TRADER_PREFLIGHT_WARN_ONLY", "1")
    return force_remote_env
