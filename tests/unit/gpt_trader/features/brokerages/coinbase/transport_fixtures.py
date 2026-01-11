from __future__ import annotations

import pytest

from gpt_trader.features.brokerages.coinbase.transports import MockTransport, NoopTransport


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
