from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock

import pytest

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

from pathlib import Path

from gpt_trader.config.runtime_settings import RuntimeSettings
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from gpt_trader.features.brokerages.coinbase.transports import MockTransport, NoopTransport
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog


@dataclass(frozen=True)
class CDPCredentials:
    api_key: str
    private_key: str
    skip_reason: str | None = None


@pytest.fixture(autouse=True)
def fast_retry_sleep(fake_clock, monkeypatch):
    """Auto-use deterministic clock so retry loops advance instantly."""
    monkeypatch.setattr("time.sleep", fake_clock.sleep)
    return fake_clock


@pytest.fixture(autouse=True)
def fast_retry_env(monkeypatch, request):
    """Ensure Coinbase client retries run with zero delay for faster tests."""
    if "test_coinbase_system.py" in request.node.nodeid:
        monkeypatch.delenv("COINBASE_FAST_RETRY", raising=False)
        yield
        return
    monkeypatch.setenv("COINBASE_FAST_RETRY", "1")
    yield


@pytest.fixture
def coinbase_cdp_credentials() -> CDPCredentials:
    """Provide Coinbase CDP API credentials or skip when unavailable."""

    if load_dotenv is not None:
        load_dotenv()

    api_key = os.getenv("COINBASE_PROD_CDP_API_KEY")
    private_key = os.getenv("COINBASE_PROD_CDP_PRIVATE_KEY")

    if not api_key or not private_key:
        skip_reason = "COINBASE_PROD_CDP_* credentials not set"
        pytest.skip(skip_reason)

    return CDPCredentials(api_key=api_key, private_key=private_key)


@pytest.fixture
def fake_clock():
    """Provides a controllable time source for testing."""
    import time

    class FakeClock:
        def __init__(self):
            self._current = time.time()

        def time(self):
            return self._current

        def sleep(self, duration):
            self._current += duration

    return FakeClock()


# =============================================================================
# MARKET DATA TEST INFRASTRUCTURE
# =============================================================================


@pytest.fixture
def mock_api_config() -> APIConfig:
    """Mock API configuration for testing."""
    return APIConfig(
        api_key="test_key",
        api_secret="test_secret",
        passphrase="test_passphrase",
        base_url="https://api.coinbase.com",
        sandbox=True,
    )


@pytest.fixture
def mock_runtime_settings() -> RuntimeSettings:
    """Mock runtime settings with WebSocket configuration."""
    return RuntimeSettings(
        raw_env={},
        runtime_root=Path("/tmp/test"),
        event_store_root_override=None,
        coinbase_default_quote="USD",
        coinbase_default_quote_overridden=False,
        coinbase_enable_derivatives=False,
        coinbase_enable_derivatives_overridden=False,
        perps_enable_streaming=False,
        perps_stream_level=1,
        perps_paper_trading=False,
        perps_force_mock=False,
        perps_skip_startup_reconcile=False,
        perps_position_fraction=None,
        order_preview_enabled=False,
        spot_force_live=False,
        broker_hint=None,
        coinbase_sandbox_enabled=True,
        coinbase_api_mode="sandbox",
        risk_config_path=None,
        coinbase_intx_portfolio_uuid=None,
    )


@pytest.fixture
def mock_product_catalog() -> ProductCatalog:
    """Mock product catalog with common trading pairs."""
    catalog = Mock(spec=ProductCatalog)
    catalog.get_symbol.return_value = "BTC-USD"
    catalog.is_supported.return_value = True
    catalog.get_quote_currency.return_value = "USD"
    return catalog


@pytest.fixture
def market_data_service() -> MarketDataService:
    """Create MarketDataService instance for testing."""
    return MarketDataService()


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


# Transport Layer Fixtures
@pytest.fixture
def mock_transport():
    """Create MockTransport with predefined messages."""
    messages = [
        {
            "type": "ticker",
            "product_id": "BTC-USD",
            "price": "50000.00",
            "bid": "49900.00",
            "ask": "50100.00",
        },
        {
            "type": "match",
            "product_id": "BTC-USD",
            "price": "50050.00",
            "size": "0.1",
            "side": "buy",
        },
        {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "49950.00", "0.5"], ["sell", "50100.00", "0.3"]],
        },
    ]
    return MockTransport(messages=messages)


@pytest.fixture
def noop_transport():
    """Create NoopTransport for testing disabled streaming."""
    return NoopTransport()


@pytest.fixture
def mock_transport_with_connection_failure():
    """Mock transport that fails to connect."""
    transport = MockTransport()
    transport.connect.side_effect = ConnectionError("Connection failed")
    return transport


# Deterministic Message Generators
@pytest.fixture
def ticker_message_factory():
    """Factory for creating ticker messages with deterministic content."""

    def create_ticker(
        symbol: str = "BTC-USD",
        price: str = "50000.00",
        bid: str = "49900.00",
        ask: str = "50100.00",
    ):
        return {
            "type": "ticker",
            "product_id": symbol,
            "price": price,
            "best_bid": bid,
            "best_ask": ask,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": 12345,
        }

    return create_ticker


@pytest.fixture
def trade_message_factory():
    """Factory for creating trade/match messages."""

    def create_trade(
        symbol: str = "BTC-USD", price: str = "50050.00", size: str = "0.1", side: str = "buy"
    ):
        return {
            "type": "match",
            "product_id": symbol,
            "price": price,
            "size": size,
            "side": side,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trade_id": "123456",
            "sequence": 12346,
        }

    return create_trade


@pytest.fixture
def orderbook_message_factory():
    """Factory for creating orderbook update messages."""

    def create_orderbook(symbol: str = "BTC-USD", changes: list | None = None):
        if changes is None:
            changes = [
                ["buy", "49950.00", "0.5"],
                ["sell", "50100.00", "0.3"],
            ]
        return {
            "type": "l2update",
            "product_id": symbol,
            "changes": changes,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": 12347,
        }

    return create_orderbook


@pytest.fixture
def heartbeat_message_factory():
    """Factory for creating heartbeat/status messages."""

    def create_heartbeat(status: str = "ok", message: str = "healthy"):
        return {
            "type": "heartbeat",
            "status": status,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return create_heartbeat


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


@pytest.fixture
def message_parsing_error_scenarios():
    """Dictionary of message parsing error scenarios."""
    return {
        "invalid_json": '{"invalid": json structure}',
        "missing_type": {"price": "50000.00", "product_id": "BTC-USD"},
        "invalid_price": {"type": "ticker", "product_id": "BTC-USD", "price": "invalid_price"},
        "missing_symbol": {"type": "ticker", "price": "50000.00"},
        "null_message": None,
        "empty_dict": {},
        "wrong_type": 12345,
    }


# Cache and Data Management Helpers
@pytest.fixture
def mock_mark_cache():
    """Mock mark cache with deterministic behavior."""
    cache = Mock()
    cache.get.return_value = Decimal("50000.00")
    cache.set.return_value = None
    cache.is_valid.return_value = True
    cache.invalidate.return_value = None
    return cache


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        "BTC-USD": {
            "bid": Decimal("49900.00"),
            "ask": Decimal("50100.00"),
            "last": Decimal("50000.00"),
            "volume": Decimal("123.45"),
            "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        },
        "ETH-USD": {
            "bid": Decimal("2990.00"),
            "ask": Decimal("3010.00"),
            "last": Decimal("3000.00"),
            "volume": Decimal("567.89"),
            "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        },
    }


# Subscription Management Helpers
@pytest.fixture
def subscription_factory():
    """Factory for creating subscription messages."""

    def create_subscription(channels: list[str], product_ids: list[str]):
        return {
            "type": "subscribe",
            "channels": channels,
            "product_ids": product_ids,
        }

    return create_subscription


@pytest.fixture
def sample_subscriptions():
    """Sample subscription configurations."""
    return {
        "ticker_only": ["ticker"],
        "full_market": ["ticker", "matches", "level2"],
        "user_events": ["user"],
        "minimal": ["heartbeat"],
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


# Time-based Testing Helpers
@pytest.fixture
def time_helpers():
    """Helper functions for time-based testing."""

    class TimeHelpers:
        @staticmethod
        def utc_now() -> datetime:
            return datetime.now(timezone.utc)

        @staticmethod
        def seconds_ago(seconds: int) -> datetime:
            return datetime.now(timezone.utc) - timedelta(seconds=seconds)

        @staticmethod
        def minutes_ago(minutes: int) -> datetime:
            return datetime.now(timezone.utc) - timedelta(minutes=minutes)

        @staticmethod
        def is_stale(timestamp: datetime, staleness_seconds: int = 30) -> bool:
            return (datetime.now(timezone.utc) - timestamp).total_seconds() > staleness_seconds

    return TimeHelpers()


# Integration Test Helpers
@pytest.fixture
def market_data_integration_setup(market_data_service, mock_transport, ticker_message_factory):
    """Complete setup for market data integration testing."""

    # Initialize symbols
    symbols = ["BTC-USD", "ETH-USD"]
    market_data_service.initialise_symbols(symbols)

    # Setup transport with test messages
    for symbol in symbols:
        message = ticker_message_factory(symbol=symbol)
        mock_transport.add_message(message)

    return {
        "service": market_data_service,
        "transport": mock_transport,
        "symbols": symbols,
        "message_count": len(mock_transport.messages),
    }
