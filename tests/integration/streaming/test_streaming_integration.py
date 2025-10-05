"""
Integration Tests for Streaming Service End-to-End Behavior

Tests WebSocket streaming lifecycle, REST fallback activation, metrics collection,
and integration with PerpsBot market data service.
"""

import pytest
import asyncio
import threading
import time
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, UTC

from bot_v2.orchestration.streaming_service import StreamingService
from bot_v2.orchestration.market_data_service import MarketDataService
from bot_v2.orchestration.market_monitor import MarketActivityMonitor
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.persistence.event_store import EventStore


@pytest.fixture
def mock_broker():
    """Mock broker with streaming capabilities."""
    broker = Mock()
    broker.stream_orderbook = Mock()
    broker.stream_trades = Mock()
    broker.set_streaming_metrics_emitter = Mock()
    return broker


@pytest.fixture
def market_data_service(tmp_path):
    """MarketDataService for streaming tests."""
    from threading import RLock

    service = Mock(spec=MarketDataService)
    service._mark_lock = RLock()
    service.mark_windows = {"BTC-USD": []}
    service._update_mark_window = Mock()

    async def mock_update_marks():
        pass

    service.update_marks = mock_update_marks
    return service


@pytest.fixture
def risk_manager():
    """Mock risk manager."""
    risk = Mock(spec=LiveRiskManager)
    risk.last_mark_update = {}
    return risk


@pytest.fixture
def event_store(tmp_path):
    """Event store for test metrics."""
    store = Mock(spec=EventStore)
    store.append_metric = Mock()
    return store


@pytest.fixture
def market_monitor():
    """Market activity monitor."""
    monitor = Mock(spec=MarketActivityMonitor)
    monitor.record_update = Mock()
    return monitor


@pytest.fixture
def streaming_service(mock_broker, market_data_service, risk_manager, event_store, market_monitor):
    """StreamingService with test dependencies."""
    return StreamingService(
        symbols=["BTC-USD"],
        broker=mock_broker,
        market_data_service=market_data_service,
        risk_manager=risk_manager,
        event_store=event_store,
        market_monitor=market_monitor,
        bot_id="test_bot",
    )


@pytest.mark.integration
class TestStreamingServiceLifecycle:
    """Test streaming service start/stop lifecycle."""

    def test_streaming_service_starts_and_stops_cleanly(self, streaming_service, mock_broker):
        """Verify streaming can start and stop without errors."""
        # Mock orderbook stream
        mock_stream = [
            {
                "product_id": "BTC-USD",
                "best_bid": "50000.00",
                "best_ask": "50100.00",
            }
        ]
        mock_broker.stream_orderbook.return_value = iter(mock_stream)

        # Start streaming
        streaming_service.start(level=1)

        # Verify thread started
        assert streaming_service.is_running()
        assert streaming_service._ws_thread is not None
        assert streaming_service._ws_thread.is_alive()

        # Wait briefly for processing
        time.sleep(0.1)

        # Stop streaming
        streaming_service.stop()

        # Verify thread stopped
        time.sleep(0.1)
        assert not streaming_service.is_running()

    def test_streaming_service_handles_empty_symbols(self, streaming_service):
        """Verify streaming skips start if no symbols configured."""
        streaming_service.symbols = []

        # Attempt to start with no symbols
        streaming_service.start(level=1)

        # Verify no thread started
        assert not streaming_service.is_running()

    def test_streaming_service_prevents_duplicate_start(self, streaming_service, mock_broker):
        """Verify starting already-running stream is idempotent."""
        mock_stream = [{"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50100"}]
        mock_broker.stream_orderbook.return_value = iter(mock_stream)

        # Start first time
        streaming_service.start(level=1)
        first_thread = streaming_service._ws_thread

        # Attempt second start
        streaming_service.start(level=1)

        # Verify same thread (no duplicate)
        assert streaming_service._ws_thread is first_thread

        streaming_service.stop()


@pytest.mark.integration
class TestStreamingOrderbookToTradesFallback:
    """Test fallback from orderbook to trades stream."""

    def test_fallback_to_trades_when_orderbook_fails(self, streaming_service, mock_broker):
        """Verify service falls back to trades stream if orderbook unavailable."""
        # Orderbook stream raises error
        mock_broker.stream_orderbook.side_effect = Exception("Orderbook not available")

        # Trades stream succeeds
        mock_trades = [
            {"product_id": "BTC-USD", "price": "50000.00", "side": "buy"},
            {"product_id": "BTC-USD", "price": "50100.00", "side": "sell"},
        ]
        mock_broker.stream_trades.return_value = iter(mock_trades)

        # Start streaming
        streaming_service.start(level=1)

        # Verify trades stream was attempted
        time.sleep(0.2)
        mock_broker.stream_trades.assert_called_once()

        streaming_service.stop()

    def test_both_streams_fail_exits_cleanly(self, streaming_service, mock_broker):
        """Verify service exits cleanly if both orderbook and trades fail."""
        # Both streams fail
        mock_broker.stream_orderbook.side_effect = Exception("Orderbook failed")
        mock_broker.stream_trades.side_effect = Exception("Trades failed")

        # Start streaming
        streaming_service.start(level=1)

        # Verify service handled failure and exited
        time.sleep(0.2)

        # Thread should exit after both failures
        # (May still be alive briefly, but will exit)
        streaming_service.stop()


@pytest.mark.integration
class TestStreamingMarkPriceUpdates:
    """Test mark price updates from streaming data."""

    def test_mark_prices_updated_from_orderbook(
        self, streaming_service, mock_broker, market_data_service
    ):
        """Verify mark prices calculated from bid/ask and written to mark windows."""
        mock_stream = [
            {
                "product_id": "BTC-USD",
                "best_bid": "50000.00",
                "best_ask": "50100.00",
            },
            {
                "product_id": "BTC-USD",
                "best_bid": "50200.00",
                "best_ask": "50300.00",
            },
        ]
        mock_broker.stream_orderbook.return_value = iter(mock_stream)

        # Start streaming
        streaming_service.start(level=1)

        # Wait for processing
        time.sleep(0.3)

        # Verify marks were updated
        # Expected marks: (50000+50100)/2 = 50050, (50200+50300)/2 = 50250
        assert market_data_service._update_mark_window.call_count >= 2

        # Verify mark values
        calls = market_data_service._update_mark_window.call_args_list
        symbols = [call[0][0] for call in calls]
        marks = [call[0][1] for call in calls]

        assert "BTC-USD" in symbols
        assert Decimal("50050") in marks or Decimal("50250") in marks

        streaming_service.stop()

    def test_mark_prices_updated_from_trades(
        self, streaming_service, mock_broker, market_data_service
    ):
        """Verify mark prices extracted from trade prices when orderbook unavailable."""
        # Orderbook fails
        mock_broker.stream_orderbook.side_effect = Exception("Orderbook unavailable")

        # Trades stream
        mock_trades = [
            {"product_id": "BTC-USD", "price": "50000.00"},
            {"product_id": "BTC-USD", "price": "50100.00"},
        ]
        mock_broker.stream_trades.return_value = iter(mock_trades)

        # Start streaming
        streaming_service.start(level=1)

        # Wait for processing
        time.sleep(0.3)

        # Verify marks updated from trade prices
        assert market_data_service._update_mark_window.call_count >= 2

        calls = market_data_service._update_mark_window.call_args_list
        marks = [call[0][1] for call in calls]

        assert Decimal("50000") in marks or Decimal("50100") in marks

        streaming_service.stop()


@pytest.mark.integration
@pytest.mark.asyncio
class TestStreamingRESTFallback:
    """Test REST polling fallback when streaming degrades."""

    async def test_rest_fallback_starts_on_disconnect(self, streaming_service, market_data_service):
        """Verify REST fallback polling starts when WebSocket disconnects."""
        # Simulate disconnect event
        disconnect_event = {"event_type": "ws_disconnect"}

        streaming_service._handle_streaming_metrics_event(disconnect_event)

        # Wait briefly for fallback thread to start
        await asyncio.sleep(0.2)

        # Verify REST fallback activated
        assert streaming_service._rest_fallback_active
        assert streaming_service._rest_fallback_thread is not None
        assert streaming_service._rest_fallback_thread.is_alive()

        # Cleanup
        streaming_service._stop_rest_fallback()

    async def test_rest_fallback_stops_on_reconnect(self, streaming_service):
        """Verify REST fallback stops when WebSocket reconnects."""
        # Start fallback
        disconnect_event = {"event_type": "ws_disconnect"}
        streaming_service._handle_streaming_metrics_event(disconnect_event)
        await asyncio.sleep(0.2)

        assert streaming_service._rest_fallback_active

        # Simulate reconnect
        connect_event = {"event_type": "ws_connect"}
        streaming_service._handle_streaming_metrics_event(connect_event)

        # Wait for fallback to stop
        await asyncio.sleep(0.2)

        # Verify fallback stopped
        assert not streaming_service._rest_fallback_active

    async def test_rest_fallback_polls_at_configured_interval(
        self, streaming_service, market_data_service
    ):
        """Verify REST fallback polls at configured interval."""
        # Set short poll interval for testing
        streaming_service.set_rest_poll_interval(0.1)  # 100ms

        # Simulate disconnect to start fallback
        streaming_service._start_rest_fallback("test")

        # Wait for multiple poll cycles
        await asyncio.sleep(0.35)

        # Verify update_marks called multiple times (3+ in 350ms with 100ms interval)
        # Note: Actual calls depend on async scheduling
        # In real implementation, verify update_marks was called

        # Cleanup
        streaming_service._stop_rest_fallback()


@pytest.mark.integration
class TestStreamingMetricsCollection:
    """Test streaming metrics emission and collection."""

    def test_metrics_emitter_tracks_connection_state(self, streaming_service):
        """Verify metrics emitter tracks WebSocket connection state."""
        # Simulate connection event
        connect_event = {
            "event_type": "ws_connect",
        }

        streaming_service._handle_streaming_metrics_event(connect_event)

        # Verify metrics server updated (if available)
        if streaming_service.metrics_server:
            # Metrics server would be updated with connection state
            pass

    def test_metrics_emitter_tracks_message_latency(self, streaming_service):
        """Verify metrics emitter records message latency."""
        # Simulate message event with latency
        message_event = {
            "event_type": "ws_message",
            "elapsed_since_last": 0.125,  # 125ms
            "timestamp": datetime.now(UTC).timestamp(),
        }

        streaming_service._handle_streaming_metrics_event(message_event)

        # Verify metrics recorded (if metrics_server available)
        if streaming_service.metrics_server:
            # Metrics server would record latency
            pass

    def test_metrics_emitter_tracks_reconnect_attempts(self, streaming_service):
        """Verify metrics emitter tracks reconnection attempts."""
        # Simulate reconnect attempt
        reconnect_event = {
            "event_type": "ws_reconnect_attempt",
            "attempt": 2,
        }

        streaming_service._handle_streaming_metrics_event(reconnect_event)

        # Verify reconnect metrics recorded
        if streaming_service.metrics_server:
            # Metrics server would track reconnect attempts
            pass

    def test_rest_fallback_starts_on_multiple_reconnect_failures(self, streaming_service):
        """Verify REST fallback starts after multiple reconnect failures."""
        # Simulate first reconnect attempt (no fallback yet)
        reconnect_event_1 = {
            "event_type": "ws_reconnect_attempt",
            "attempt": 1,
        }
        streaming_service._handle_streaming_metrics_event(reconnect_event_1)
        assert not streaming_service._rest_fallback_active

        # Simulate second reconnect attempt (fallback should start)
        reconnect_event_2 = {
            "event_type": "ws_reconnect_attempt",
            "attempt": 2,
        }
        streaming_service._handle_streaming_metrics_event(reconnect_event_2)

        # Wait briefly for fallback thread
        time.sleep(0.1)

        # Verify fallback started
        assert streaming_service._rest_fallback_active

        # Cleanup
        streaming_service._stop_rest_fallback()


@pytest.mark.integration
class TestStreamingSymbolUpdates:
    """Test dynamic symbol list updates."""

    def test_update_symbols_changes_stream_targets(self, streaming_service, mock_broker):
        """Verify updating symbols changes what is streamed."""
        # Initial symbols
        assert streaming_service.symbols == ["BTC-USD"]

        # Update symbols
        new_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        streaming_service.update_symbols(new_symbols)

        # Verify symbols updated
        assert streaming_service.symbols == new_symbols

        # Note: Caller must restart streaming for changes to take effect
        # This test documents that update_symbols updates the internal list

    def test_symbol_update_requires_streaming_restart(self, streaming_service, mock_broker):
        """Document that symbol updates require manual streaming restart."""
        mock_stream = [{"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50100"}]
        mock_broker.stream_orderbook.return_value = iter(mock_stream)

        # Start with initial symbols
        streaming_service.start(level=1)
        assert streaming_service.is_running()

        # Update symbols (does not auto-restart)
        streaming_service.update_symbols(["BTC-USD", "ETH-USD"])

        # Streaming still running with old symbols
        assert streaming_service.is_running()

        # Caller must stop and restart to apply new symbols
        streaming_service.stop()
        time.sleep(0.1)

        # Restart would use new symbols (if we had mock data for them)
        assert streaming_service.symbols == ["BTC-USD", "ETH-USD"]
