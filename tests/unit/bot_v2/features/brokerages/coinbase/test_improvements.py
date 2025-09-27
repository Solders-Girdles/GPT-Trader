"""Unit tests for Coinbase integration improvements."""

import time
from unittest.mock import MagicMock, patch
from decimal import Decimal

import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient, CoinbaseAuth
from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce


def test_rate_limit_tracking():
    """Test that rate limit tracking is initialized correctly."""
    client = CoinbaseClient(base_url="https://api.coinbase.com")
    
    # Verify rate limit tracking attributes exist
    assert hasattr(client, '_request_count')
    assert hasattr(client, '_request_window_start')
    assert client._request_count == 0
    assert client._request_window_start > 0  # Should be initialized with current time
    
    # Mock the transport to avoid actual network calls
    def mock_transport(method, url, headers, body, timeout):
        return (200, {}, '{"success": true}')
    
    client._transport = mock_transport
    
    # Make a request and verify counter increments
    initial_count = client._request_count
    client._request("GET", "/test")
    assert client._request_count == initial_count + 1


def test_connection_validation():
    """Test that connection validation checks accounts."""
    config = APIConfig(
        api_key="test_key",
        api_secret="test_secret",
        passphrase=None,
        base_url="https://api.coinbase.com",
        sandbox=False
    )
    
    broker = CoinbaseBrokerage(config)
    
    # Mock the client's get_accounts method
    mock_accounts_response = {
        "accounts": [
            {"uuid": "test-account-123", "currency": "USD", "balance": "100.00"}
        ]
    }
    
    # Use public client attribute (adapter exposes `client`)
    broker.client.get_accounts = MagicMock(return_value=mock_accounts_response)
    
    # Test connection
    result = broker.connect()
    assert result is True
    assert broker._connected is True
    assert broker._account_id == "test-account-123"


def test_position_list_spot_trading():
    """Test that list_positions returns empty for spot trading."""
    config = APIConfig(
        api_key="test_key",
        api_secret="test_secret",
        passphrase=None,
        base_url="https://api.coinbase.com",
        sandbox=False,
        enable_derivatives=False  # Spot trading only
    )
    
    broker = CoinbaseBrokerage(config)
    
    # Should return empty list for spot trading
    positions = broker.list_positions()
    assert positions == []
    
    # Should not have called CFM endpoints
    broker.client.cfm_positions = MagicMock()
    positions = broker.list_positions()
    assert positions == []
    broker.client.cfm_positions.assert_not_called()


def test_order_error_handling():
    """Test that order placement handles errors correctly."""
    config = APIConfig(
        api_key="test_key",
        api_secret="test_secret",
        passphrase=None,
        base_url="https://api.coinbase.com",
        sandbox=False
    )
    
    broker = CoinbaseBrokerage(config)
    
    # Mock product catalog to avoid actual API calls
    mock_product = MagicMock()
    mock_product.step_size = Decimal("0.001")
    mock_product.price_increment = Decimal("0.01")
    mock_product.min_size = Decimal("0.001")
    mock_product.min_notional = None
    
    broker.product_catalog.get = MagicMock(return_value=mock_product)
    
    # Test insufficient funds error
    from bot_v2.features.brokerages.core.interfaces import InsufficientFunds
    broker.client.place_order = MagicMock(side_effect=InsufficientFunds("Not enough balance"))
    
    with pytest.raises(InsufficientFunds):
        broker.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            qty=Decimal("1.0")
        )
