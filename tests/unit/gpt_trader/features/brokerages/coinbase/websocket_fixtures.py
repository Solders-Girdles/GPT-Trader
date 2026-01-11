from __future__ import annotations

from unittest.mock import Mock

import pytest


# WebSocket Connection Scenarios
@pytest.fixture
def mock_websocket_connected():
    """Mock WebSocket in connected state."""
    ws = Mock()
    ws.connected = True
    ws.ping.return_value = None
    return ws


@pytest.fixture
def mock_websocket_disconnected():
    """Mock WebSocket in disconnected state."""
    ws = Mock()
    ws.connected = False
    ws.ping.side_effect = ConnectionError("WebSocket not connected")
    return ws


@pytest.fixture
def mock_websocket_auth_failure():
    """Mock WebSocket that fails authentication."""
    ws = Mock()
    ws.connect.side_effect = Exception("Authentication failed")
    return ws


@pytest.fixture
def mock_websocket_with_reconnect_backoff():
    """Mock WebSocket that triggers reconnection with exponential backoff."""
    ws = Mock()
    call_count = 0

    def connect_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise ConnectionError(f"Connection attempt {call_count} failed")
        return None

    ws.connect.side_effect = connect_side_effect
    ws.connected = False
    return ws


# Error Scenario Helpers
@pytest.fixture
def connection_error_scenarios():
    """Dictionary of connection error scenarios for testing."""
    return {
        "authentication_failed": Exception("Invalid API credentials"),
        "rate_limited": Exception("Rate limit exceeded"),
        "network_timeout": TimeoutError("Connection timeout"),
        "server_error": ConnectionError("Server unavailable"),
        "ssl_error": ConnectionError("SSL verification failed"),
    }


# WebSocket Lifecycle Helpers
@pytest.fixture
def websocket_lifecycle_states():
    """Dictionary of WebSocket lifecycle states for testing."""
    return {
        "connecting": {"connected": False, "connecting": True, "authenticated": False},
        "connected": {"connected": True, "connecting": False, "authenticated": False},
        "authenticated": {"connected": True, "connecting": False, "authenticated": True},
        "disconnected": {"connected": False, "connecting": False, "authenticated": False},
        "error": {
            "connected": False,
            "connecting": False,
            "authenticated": False,
            "error": "Connection failed",
        },
    }
