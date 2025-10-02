"""
Comprehensive tests for CoinbaseWebSocketHandler.

Covers message processing, normalization, error handling, and market data updates.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.brokerages.coinbase.websocket_handler import CoinbaseWebSocketHandler
from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket, WSSubscription


@pytest.fixture
def mock_endpoints():
    """Create mock endpoints."""
    endpoints = Mock()
    endpoints.websocket_url = Mock(return_value="wss://test.coinbase.com")
    return endpoints


@pytest.fixture
def mock_config():
    """Create mock API config."""
    config = Mock()
    config.api_key = "test-key"
    config.api_secret = "test-secret"
    return config


@pytest.fixture
def mock_market_data():
    """Create mock MarketDataService."""
    market_data = Mock()
    market_data.initialise_symbols = Mock()
    market_data.has_symbol = Mock(return_value=True)
    market_data.update_ticker = Mock()
    market_data.record_trade = Mock()
    market_data.update_depth = Mock()
    market_data.set_mark = Mock()
    return market_data


@pytest.fixture
def mock_rest_service():
    """Create mock REST service."""
    rest = Mock()
    rest.update_position_metrics = Mock()
    rest.process_fill_for_pnl = Mock()
    return rest


@pytest.fixture
def mock_product_catalog():
    """Create mock ProductCatalog."""
    catalog = Mock()
    return catalog


@pytest.fixture
def mock_ws_client():
    """Create mock WebSocket client."""
    ws = Mock(spec=CoinbaseWebSocket)
    ws.subscribe = Mock()
    ws.on_message = None
    ws.stream_messages = Mock(return_value=iter([]))
    return ws


@pytest.fixture
def handler(mock_endpoints, mock_config, mock_market_data, mock_rest_service, mock_product_catalog):
    """Create handler with mocked dependencies."""
    return CoinbaseWebSocketHandler(
        endpoints=mock_endpoints,
        config=mock_config,
        market_data=mock_market_data,
        rest_service=mock_rest_service,
        product_catalog=mock_product_catalog,
        client_auth=None,
    )


class TestStartMarketData:
    """Test start_market_data method."""

    def test_start_market_data_basic(self, handler, mock_market_data, mock_ws_client):
        """Should initialize symbols and subscribe to channels."""
        handler._ws_factory_override = lambda: mock_ws_client

        handler.start_market_data(["BTC-USD", "ETH-USD"])

        # Should initialize symbols
        mock_market_data.initialise_symbols.assert_called_once()
        symbols = mock_market_data.initialise_symbols.call_args[0][0]
        assert "BTC-USD" in symbols

        # Should subscribe to 3 channels
        assert mock_ws_client.subscribe.call_count == 3

    def test_start_market_data_empty_symbols(self, handler, mock_market_data):
        """Should do nothing with empty symbols list."""
        handler.start_market_data([])

        mock_market_data.initialise_symbols.assert_not_called()

    def test_start_market_data_normalizes_symbols(self, handler, mock_market_data, mock_ws_client):
        """Should normalize symbols."""
        handler._ws_factory_override = lambda: mock_ws_client

        handler.start_market_data(["btc-usd"])

        symbols = mock_market_data.initialise_symbols.call_args[0][0]
        assert symbols[0] == "BTC-USD"  # Normalized

    def test_start_market_data_sets_on_message_handler(self, handler, mock_ws_client):
        """Should set on_message callback."""
        handler._ws_factory_override = lambda: mock_ws_client

        handler.start_market_data(["BTC-USD"])

        assert mock_ws_client.on_message is not None


class TestStreamTrades:
    """Test stream_trades method."""

    def test_stream_trades_basic(self, handler, mock_market_data):
        """Should stream and normalize trade messages."""
        messages = [
            {"type": "trade", "product_id": "BTC-USD", "price": "50000", "size": "0.1"},
            {"type": "trade", "product_id": "ETH-USD", "price": "3000", "size": "1.0"},
        ]

        mock_ws = Mock()
        mock_ws.subscribe = Mock()
        mock_ws.stream_messages = Mock(return_value=iter(messages))

        handler._ws_factory_override = lambda: mock_ws

        trades = list(handler.stream_trades(["BTC-USD"]))

        assert len(trades) == 2
        assert trades[0]["price"] == Decimal("50000")

    def test_stream_trades_updates_mark_price(self, handler, mock_market_data, mock_rest_service):
        """Should update mark price for valid trades."""
        messages = [
            {"type": "trade", "product_id": "BTC-USD", "price": "50000", "size": "0.1"},
        ]

        mock_ws = Mock()
        mock_ws.subscribe = Mock()
        mock_ws.stream_messages = Mock(return_value=iter(messages))

        handler._ws_factory_override = lambda: mock_ws

        list(handler.stream_trades(["BTC-USD"]))

        # Should set mark price
        mock_market_data.set_mark.assert_called_with("BTC-USD", Decimal("50000"))
        mock_rest_service.update_position_metrics.assert_called_with("BTC-USD")

    def test_stream_trades_uses_provided_ws(self, handler):
        """Should use provided WebSocket client."""
        messages = [{"type": "trade", "product_id": "BTC-USD", "price": "50000"}]

        mock_ws = Mock()
        mock_ws.subscribe = Mock()
        mock_ws.stream_messages = Mock(return_value=iter(messages))

        list(handler.stream_trades(["BTC-USD"], ws=mock_ws))

        mock_ws.subscribe.assert_called_once()


class TestStreamOrderbook:
    """Test stream_orderbook method."""

    def test_stream_orderbook_level1(self, handler):
        """Should use ticker channel for level 1."""
        messages = [
            {"type": "ticker", "product_id": "BTC-USD", "bid": "49999", "ask": "50001"},
        ]

        mock_ws = Mock()
        mock_ws.subscribe = Mock()
        mock_ws.stream_messages = Mock(return_value=iter(messages))

        handler._ws_factory_override = lambda: mock_ws

        list(handler.stream_orderbook(["BTC-USD"], level=1))

        # Should subscribe to ticker
        sub = mock_ws.subscribe.call_args[0][0]
        assert "ticker" in sub.channels

    def test_stream_orderbook_level2(self, handler):
        """Should use level2 channel for level >= 2."""
        messages = []

        mock_ws = Mock()
        mock_ws.subscribe = Mock()
        mock_ws.stream_messages = Mock(return_value=iter(messages))

        handler._ws_factory_override = lambda: mock_ws

        list(handler.stream_orderbook(["BTC-USD"], level=2))

        # Should subscribe to level2
        sub = mock_ws.subscribe.call_args[0][0]
        assert "level2" in sub.channels

    def test_stream_orderbook_updates_mark_from_ticker(self, handler, mock_market_data, mock_rest_service):
        """Should calculate and set mark price from bid/ask."""
        messages = [
            {"type": "ticker", "product_id": "BTC-USD", "best_bid": "49999", "best_ask": "50001"},
        ]

        mock_ws = Mock()
        mock_ws.subscribe = Mock()
        mock_ws.stream_messages = Mock(return_value=iter(messages))

        handler._ws_factory_override = lambda: mock_ws

        list(handler.stream_orderbook(["BTC-USD"], level=1))

        # Should set mark to mid price
        expected_mid = (Decimal("49999") + Decimal("50001")) / 2
        mock_market_data.set_mark.assert_called_with("BTC-USD", expected_mid)


class TestStreamUserEvents:
    """Test stream_user_events method."""

    def test_stream_user_events_basic(self, handler):
        """Should stream user events with sequence annotation."""
        messages = [
            {"type": "order", "sequence": 100},
            {"type": "fill", "sequence": 101},
        ]

        mock_ws = Mock()
        mock_ws.subscribe = Mock()
        mock_ws.stream_messages = Mock(return_value=iter(messages))

        handler._ws_factory_override = lambda: mock_ws

        events = list(handler.stream_user_events())

        assert len(events) == 2
        assert "sequence" in events[0]

    def test_stream_user_events_processes_fills(self, handler, mock_rest_service):
        """Should process fill events for PnL."""
        messages = [
            {"type": "fill", "sequence": 101, "product_id": "BTC-USD"},
        ]

        mock_ws = Mock()
        mock_ws.subscribe = Mock()
        mock_ws.stream_messages = Mock(return_value=iter(messages))

        handler._ws_factory_override = lambda: mock_ws

        list(handler.stream_user_events())

        # Should process fill
        mock_rest_service.process_fill_for_pnl.assert_called_once()

    def test_stream_user_events_with_product_ids(self, handler):
        """Should subscribe with specific product IDs."""
        mock_ws = Mock()
        mock_ws.subscribe = Mock()
        mock_ws.stream_messages = Mock(return_value=iter([]))

        handler._ws_factory_override = lambda: mock_ws

        list(handler.stream_user_events(product_ids=["BTC-USD", "ETH-USD"]))

        # Should include product IDs in subscription
        sub = mock_ws.subscribe.call_args[0][0]
        assert "BTC-USD" in sub.product_ids


class TestHandleWSMessage:
    """Test _handle_ws_message method."""

    @patch('bot_v2.features.brokerages.coinbase.websocket_handler.datetime')
    def test_handle_ticker_message(self, mock_datetime, handler, mock_market_data):
        """Should process ticker messages and update market data."""
        mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)

        message = {
            "type": "ticker",
            "product_id": "BTC-USD",
            "best_bid": "49999.50",
            "best_ask": "50000.50",
            "price": "50000.00",
        }

        handler._handle_ws_message(message)

        # Should update ticker
        mock_market_data.update_ticker.assert_called_once()
        call_args = mock_market_data.update_ticker.call_args[0]
        assert call_args[0] == "BTC-USD"
        assert call_args[1] == Decimal("49999.50")  # bid
        assert call_args[2] == Decimal("50000.50")  # ask
        assert call_args[3] == Decimal("50000.00")  # last

    @patch('bot_v2.features.brokerages.coinbase.websocket_handler.datetime')
    def test_handle_match_message(self, mock_datetime, handler, mock_market_data):
        """Should process match messages and record trades."""
        mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)

        message = {
            "type": "match",
            "product_id": "BTC-USD",
            "size": "0.5",
        }

        handler._handle_ws_message(message)

        # Should record trade
        mock_market_data.record_trade.assert_called_once()
        call_args = mock_market_data.record_trade.call_args[0]
        assert call_args[0] == "BTC-USD"
        assert call_args[1] == Decimal("0.5")

    @patch('bot_v2.features.brokerages.coinbase.websocket_handler.datetime')
    def test_handle_l2update_message(self, mock_datetime, handler, mock_market_data):
        """Should process level2 updates."""
        mock_datetime.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)

        message = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "changes": [
                ["buy", "49999.00", "1.5"],
                ["sell", "50001.00", "2.0"],
            ],
        }

        handler._handle_ws_message(message)

        # Should update depth
        mock_market_data.update_depth.assert_called_once()
        call_args = mock_market_data.update_depth.call_args[0]
        assert call_args[0] == "BTC-USD"
        assert len(call_args[1]) == 2

    def test_handle_message_unknown_symbol(self, handler, mock_market_data):
        """Should ignore messages for unknown symbols."""
        mock_market_data.has_symbol.return_value = False

        message = {"type": "ticker", "product_id": "UNKNOWN-USD"}

        handler._handle_ws_message(message)

        # Should not update anything
        mock_market_data.update_ticker.assert_not_called()

    def test_handle_message_error(self, handler, mock_market_data):
        """Should handle errors in message processing gracefully."""
        mock_market_data.update_ticker.side_effect = Exception("Processing error")

        message = {"type": "ticker", "product_id": "BTC-USD", "best_bid": "50000"}

        # Should not crash
        handler._handle_ws_message(message)

    def test_handle_message_missing_product_id(self, handler, mock_market_data):
        """Should ignore messages without product_id."""
        message = {"type": "ticker", "price": "50000"}

        handler._handle_ws_message(message)

        mock_market_data.update_ticker.assert_not_called()


class TestNormalizeWSMessage:
    """Test _normalize_ws_message method."""

    def test_normalize_ticker_message(self, handler):
        """Should normalize ticker messages."""
        message = {
            "type": "ticker",
            "product_id": "BTC-USD",
            "last_price": "50000",
            "bid_price": "49999",
            "ask_price": "50001",
        }

        normalized = handler._normalize_ws_message(message)

        assert normalized["type"] == "ticker"
        assert normalized["price"] == "50000"
        assert normalized["best_bid"] == "49999"
        assert normalized["best_ask"] == "50001"

    def test_normalize_ticker_with_variants(self, handler):
        """Should recognize ticker channel variants."""
        variants = ["ticker", "tickers", "ticker_batch", "tick"]

        for variant in variants:
            message = {"type": variant, "product_id": "BTC-USD"}
            normalized = handler._normalize_ws_message(message)
            assert normalized["type"] == "ticker"

    def test_normalize_match_message(self, handler):
        """Should normalize match/trade messages."""
        message = {
            "type": "trade",
            "product_id": "BTC-USD",
            "quantity": "0.5",
            "trade_price": "50000",
        }

        normalized = handler._normalize_ws_message(message)

        assert normalized["type"] == "match"
        assert normalized["size"] == "0.5"
        assert normalized["price"] == "50000"

    def test_normalize_match_with_variants(self, handler):
        """Should recognize match channel variants."""
        variants = ["match", "matches", "trade", "trades", "executed_trade"]

        for variant in variants:
            message = {"type": variant, "product_id": "BTC-USD"}
            normalized = handler._normalize_ws_message(message)
            assert normalized["type"] == "match"

    def test_normalize_l2update_message(self, handler):
        """Should normalize level2 messages."""
        message = {
            "type": "l2update",
            "product_id": "BTC-USD",
            "updates": [["buy", "49999", "1.0"]],
        }

        normalized = handler._normalize_ws_message(message)

        assert normalized["type"] == "l2update"
        assert "changes" in normalized

    def test_normalize_l2update_with_bids_asks(self, handler):
        """Should convert bids/asks to changes format."""
        message = {
            "type": "level2",
            "product_id": "BTC-USD",
            "bids": [[49999, 1.0], [49998, 2.0]],
            "asks": [[50001, 1.5], [50002, 2.5]],
        }

        normalized = handler._normalize_ws_message(message)

        assert normalized["type"] == "l2update"
        assert "changes" in normalized
        # Should convert bids/asks to change format
        assert len(normalized["changes"]) > 0

    def test_normalize_l2update_with_variants(self, handler):
        """Should recognize l2update channel variants."""
        variants = ["l2update", "level2", "l2", "level2_batch", "orderbook", "book"]

        for variant in variants:
            message = {"type": variant, "product_id": "BTC-USD"}
            normalized = handler._normalize_ws_message(message)
            assert normalized["type"] == "l2update"

    def test_normalize_unknown_channel(self, handler):
        """Should return None for unknown channel types."""
        message = {"type": "unknown_channel", "product_id": "BTC-USD"}

        normalized = handler._normalize_ws_message(message)

        assert normalized is None

    def test_normalize_empty_message(self, handler):
        """Should return None for empty message."""
        normalized = handler._normalize_ws_message({})

        assert normalized is None

    def test_normalize_adds_product_id(self, handler):
        """Should add product_id from symbol or instrument."""
        message = {"type": "ticker", "symbol": "BTC-USD"}

        normalized = handler._normalize_ws_message(message)

        assert normalized["product_id"] == "BTC-USD"

    def test_normalize_l2update_limits_changes(self, handler):
        """Should limit bids/asks to 10 each."""
        bids = [[50000 - i, 1.0] for i in range(20)]
        asks = [[50000 + i, 1.0] for i in range(20)]

        message = {
            "type": "level2",
            "product_id": "BTC-USD",
            "bids": bids,
            "asks": asks,
        }

        normalized = handler._normalize_ws_message(message)

        # Should limit to 10 bids + 10 asks = 20 total
        assert len(normalized["changes"]) <= 20


class TestWSClientManagement:
    """Test WebSocket client management."""

    def test_ensure_ws_client_creates_once(self, handler):
        """Should create client only once."""
        mock_ws = Mock()
        handler._ws_factory_override = Mock(return_value=mock_ws)

        ws1 = handler._ensure_ws_client()
        ws2 = handler._ensure_ws_client()

        assert ws1 is ws2
        handler._ws_factory_override.assert_called_once()

    def test_set_ws_factory_clears_client(self, handler):
        """Should clear existing client when setting factory."""
        mock_ws = Mock()
        handler._ws_factory_override = lambda: mock_ws
        handler._ensure_ws_client()

        # Set new factory
        new_factory = Mock()
        handler.set_ws_factory_for_testing(new_factory)

        # Should have cleared client
        assert handler._ws_client is None

    def test_create_ws_uses_factory_override(self, handler):
        """Should use factory override if set."""
        mock_ws = Mock()
        handler._ws_factory_override = Mock(return_value=mock_ws)

        ws = handler._create_ws_instance()

        assert ws is mock_ws

    @patch('bot_v2.features.brokerages.coinbase.websocket_handler.build_ws_auth_provider')
    def test_create_ws_builds_auth(self, mock_build_auth, handler, mock_config):
        """Should build auth provider for WebSocket."""
        mock_build_auth.return_value = Mock()

        handler._create_ws_instance()

        mock_build_auth.assert_called_once_with(mock_config, None)

    def test_create_ws_method(self, handler):
        """Should create WebSocket instance."""
        mock_ws = Mock()
        handler._ws_factory_override = lambda: mock_ws

        ws = handler.create_ws()

        assert ws is mock_ws
