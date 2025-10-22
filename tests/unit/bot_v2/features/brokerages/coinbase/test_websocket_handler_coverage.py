"""
Focused WebSocket Handler coverage tests for 80%+ coverage improvement.

This test suite targets the CoinbaseWebSocketHandler class which is critical
for message processing, subscription management, and market data lifecycle.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.transports import MockTransport
from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
from bot_v2.features.brokerages.coinbase.websocket_handler import CoinbaseWebSocketHandler
from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket, WSSubscription


class TestWebSocketHandlerCoverage:
    """Core WebSocket Handler functionality tests for maximum coverage impact."""

    def test_handler_initialization(self, mock_api_config, mock_runtime_settings):
        """Test WebSocket handler initialization."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        assert handler._config is mock_api_config
        assert handler._market_data is market_data
        assert handler._rest_service is rest_service
        assert handler._product_catalog is product_catalog
        assert handler._client_auth is None
        assert handler._settings is mock_runtime_settings
        assert handler._ws_client is None
        assert handler._ws_factory_override is None

    def test_handler_initialization_with_custom_websocket_class(self, mock_api_config):
        """Test handler initialization with custom WebSocket class."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        custom_ws_class = Mock()

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            ws_cls=custom_ws_class,
        )

        assert handler._ws_cls is custom_ws_class

    def test_start_market_data_with_symbols(self, mock_api_config, mock_runtime_settings):
        """Test starting market data with symbols."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        # Mock WebSocket client
        mock_ws = Mock()
        handler._ws_factory_override = Mock(return_value=mock_ws)

        symbols = ["BTC-USD", "ETH-USD"]
        handler.start_market_data(symbols)

        # Should initialize symbols
        market_data.initialise_symbols.assert_called_once()
        assert mock_ws.on_message is not None

    def test_start_market_data_empty_symbols(self, mock_api_config, mock_runtime_settings):
        """Test starting market data with empty symbols list."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        # Should return early without initializing
        handler.start_market_data([])

        market_data.initialise_symbols.assert_not_called()

    def test_create_ws_instance(self, mock_api_config, mock_runtime_settings):
        """Test creating WebSocket instance."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        with patch.object(handler, '_create_ws_instance') as mock_create:
            mock_ws = Mock()
            mock_create.return_value = mock_ws

            result = handler.create_ws()

            mock_create.assert_called_once()
            assert result is mock_ws

    def test_create_ws_with_factory_override(self, mock_api_config, mock_runtime_settings):
        """Test creating WebSocket with factory override."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        mock_ws = Mock()
        handler._ws_factory_override = Mock(return_value=mock_ws)

        result = handler.create_ws()

        assert result is mock_ws
        handler._ws_factory_override.assert_called_once()

    def test_set_ws_factory_for_testing(self, mock_api_config, mock_runtime_settings):
        """Test setting WebSocket factory for testing."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        # Set up existing client
        existing_client = Mock()
        handler._ws_client = existing_client

        mock_factory = Mock()
        handler.set_ws_factory_for_testing(mock_factory)

        assert handler._ws_factory_override is mock_factory
        assert handler._ws_client is None

    def test_configure_websocket(self, mock_api_config, mock_runtime_settings):
        """Test configuring WebSocket settings."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        custom_ws_class = Mock()
        new_auth = Mock()

        handler.configure_websocket(ws_cls=custom_ws_class, client_auth=new_auth)

        assert handler._ws_cls is custom_ws_class
        assert handler._client_auth is new_auth

    def test_configure_websocket_no_changes(self, mock_api_config, mock_runtime_settings):
        """Test configuring WebSocket with no changes."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        # Set up existing client
        existing_client = Mock()
        handler._ws_client = existing_client

        # Configure with same values - should not clear existing client
        handler.configure_websocket(ws_cls=handler._ws_cls, client_auth=handler._client_auth)
        assert handler._ws_client is existing_client

    def test_set_product_catalog(self, mock_api_config, mock_runtime_settings):
        """Test setting product catalog."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        new_catalog = Mock(spec=ProductCatalog)
        handler.set_product_catalog(new_catalog)

        assert handler._product_catalog is new_catalog

    def test_ensure_ws_client_creates_new(self, mock_api_config, mock_runtime_settings):
        """Test _ensure_ws_client creates new client when none exists."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        with patch.object(handler, '_create_ws_instance') as mock_create:
            mock_ws = Mock()
            mock_create.return_value = mock_ws

            result = handler._ensure_ws_client()

            mock_create.assert_called_once()
            assert result is mock_ws
            assert handler._ws_client is mock_ws

    def test_ensure_ws_client_returns_existing(self, mock_api_config, mock_runtime_settings):
        """Test _ensure_ws_client returns existing client."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        existing_client = Mock()
        handler._ws_client = existing_client

        with patch.object(handler, '_create_ws_instance') as mock_create:
            result = handler._ensure_ws_client()

            mock_create.assert_not_called()
            assert result is existing_client

    def test_stream_trades(self, mock_api_config, mock_runtime_settings):
        """Test streaming trades."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        with patch.object(handler, '_create_ws_instance') as mock_create:
            mock_ws = Mock()
            mock_messages = [
                {"type": "match", "product_id": "BTC-USD", "price": "50000.00", "size": "0.1"},
                {"type": "match", "product_id": "ETH-USD", "price": "3000.00", "size": "1.0"},
            ]
            mock_ws.stream_messages.return_value = iter(mock_messages)
            mock_create.return_value = mock_ws

            symbols = ["BTC-USD", "ETH-USD"]
            trades = list(handler.stream_trades(symbols))

            assert len(trades) == 2
            mock_ws.subscribe.assert_called_once()

    def test_stream_orderbook(self, mock_api_config, mock_runtime_settings):
        """Test streaming orderbook."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        with patch.object(handler, '_create_ws_instance') as mock_create:
            mock_ws = Mock()
            mock_messages = [
                {"type": "ticker", "product_id": "BTC-USD", "best_bid": "49900.00", "best_ask": "50100.00"},
                {"type": "ticker", "product_id": "ETH-USD", "best_bid": "2990.00", "best_ask": "3010.00"},
            ]
            mock_ws.stream_messages.return_value = iter(mock_messages)
            mock_create.return_value = mock_ws

            symbols = ["BTC-USD", "ETH-USD"]
            orderbook = list(handler.stream_orderbook(symbols))

            assert len(orderbook) == 2
            mock_ws.subscribe.assert_called_once()

    def test_stream_user_events(self, mock_api_config, mock_runtime_settings):
        """Test streaming user events."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        with patch.object(handler, '_create_ws_instance') as mock_create:
            mock_ws = Mock()
            mock_messages = [
                {"type": "fill", "product_id": "BTC-USD", "size": "0.1", "price": "50000.00"},
                {"type": "order", "product_id": "ETH-USD", "size": "1.0", "price": "3000.00"},
            ]
            mock_ws.stream_messages.return_value = iter(mock_messages)
            mock_create.return_value = mock_ws

            events = list(handler.stream_user_events())

            assert len(events) == 2
            mock_ws.subscribe.assert_called_once()
            # Should process fill events
            rest_service.process_fill_for_pnl.assert_called_once()

    def test_handle_ws_message_ticker(self, mock_api_config, mock_runtime_settings):
        """Test handling WebSocket ticker messages."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        # Mock market data to return true for has_symbol
        market_data.has_symbol.return_value = True

        message = {
            "type": "ticker",
            "product_id": "BTC-USD",
            "best_bid": "49900.00",
            "best_ask": "50100.00",
            "price": "50000.00",
        }

        handler._handle_ws_message(message)

        market_data.update_ticker.assert_called_once()
        call_args = market_data.update_ticker.call_args[0]
        assert call_args[0] == "BTC-USD"
        assert call_args[1] == Decimal("49900.00")  # bid
        assert call_args[2] == Decimal("50100.00")  # ask
        assert call_args[3] == Decimal("50000.00")  # last

    def test_handle_ws_message_match(self, mock_api_config, mock_runtime_settings):
        """Test handling WebSocket match messages."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        market_data.has_symbol.return_value = True

        message = {
            "type": "match",
            "product_id": "BTC-USD",
            "size": "0.1",
            "price": "50000.00",
        }

        handler._handle_ws_message(message)

        market_data.record_trade.assert_called_once()
        call_args = market_data.record_trade.call_args[0]
        assert call_args[0] == "BTC-USD"
        assert call_args[1] == Decimal("0.1")
        assert isinstance(call_args[2], datetime)

    def test_handle_ws_message_l2update(self, mock_api_config, mock_runtime_settings):
        """Test handling WebSocket l2update messages."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        market_data.has_symbol.return_value = True

        message = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [["buy", "49950.00", "0.5"], ["sell", "50100.00", "0.3"]],
        }

        handler._handle_ws_message(message)

        market_data.update_depth.assert_called_once_with("BTC-USD", [["buy", "49950.00", "0.5"], ["sell", "50100.00", "0.3"]])

    def test_handle_ws_message_invalid_symbol(self, mock_api_config, mock_runtime_settings):
        """Test handling WebSocket message with invalid symbol."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        market_data.has_symbol.return_value = False

        message = {
            "type": "ticker",
            "product_id": "UNKNOWN-USD",
            "best_bid": "49900.00",
            "best_ask": "50100.00",
        }

        handler._handle_ws_message(message)

        market_data.update_ticker.assert_not_called()

    def test_handle_ws_message_error_handling(self, mock_api_config, mock_runtime_settings, caplog):
        """Test error handling in WebSocket message processing."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        market_data.has_symbol.return_value = True
        market_data.update_ticker.side_effect = Exception("Test error")

        message = {"type": "ticker", "product_id": "BTC-USD"}

        # Should not raise exception
        handler._handle_ws_message(message)

        # Should log error
        assert "Error handling WebSocket message" in caplog.text

    def test_normalize_ws_message_ticker(self, mock_api_config, mock_runtime_settings):
        """Test normalizing WebSocket ticker messages."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        # Test ticker normalization
        message = {"type": "ticker", "symbol": "BTC-USD", "last_price": "50000.00"}
        normalized = handler._normalize_ws_message(message)

        assert normalized["type"] == "ticker"
        assert normalized["product_id"] == "BTC-USD"
        assert normalized["price"] == "50000.00"

    def test_normalize_ws_message_match(self, mock_api_config, mock_runtime_settings):
        """Test normalizing WebSocket match messages."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        # Test match normalization
        message = {"channel": "trade", "instrument": "BTC-USD", "quantity": "0.1"}
        normalized = handler._normalize_ws_message(message)

        assert normalized["type"] == "match"
        assert normalized["product_id"] == "BTC-USD"
        assert normalized["size"] == "0.1"

    def test_normalize_ws_message_l2update(self, mock_api_config, mock_runtime_settings):
        """Test normalizing WebSocket l2update messages."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        # Test l2update normalization with bids/asks
        message = {
            "type": "orderbook",
            "symbol": "BTC-USD",
            "bids": [["49950.00", "0.5"], ["49940.00", "0.3"]],
            "asks": [["50100.00", "0.2"], ["50110.00", "0.4"]],
        }
        normalized = handler._normalize_ws_message(message)

        assert normalized["type"] == "l2update"
        assert normalized["product_id"] == "BTC-USD"
        assert len(normalized["changes"]) == 4

    def test_normalize_ws_message_invalid(self, mock_api_config, mock_runtime_settings):
        """Test normalizing invalid WebSocket messages."""
        market_data = Mock(spec=MarketDataService)
        rest_service = Mock()
        product_catalog = Mock(spec=ProductCatalog)

        handler = CoinbaseWebSocketHandler(
            endpoints=Mock(),
            config=mock_api_config,
            market_data=market_data,
            rest_service=rest_service,
            product_catalog=product_catalog,
            client_auth=None,
            settings=mock_runtime_settings,
        )

        # Test invalid message
        result = handler._normalize_ws_message(None)
        assert result is None

        # Test unknown message type
        result = handler._normalize_ws_message({"type": "unknown"})
        assert result is None