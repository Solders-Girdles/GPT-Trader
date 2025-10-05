"""Unit tests for StreamingService.

Tests cover:
- Start/stop semantics and thread lifecycle
- Orderbook → trades stream fallback
- Mark window updates via MarketDataService
- Event store metric writes
- Market monitor integration
- Stop event handling
"""

from __future__ import annotations

import threading
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from bot_v2.orchestration.streaming_service import StreamingService

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mock_broker():
    """Create mock broker with streaming methods."""
    broker = Mock()
    broker.stream_orderbook = Mock()
    broker.stream_trades = Mock()
    broker.set_streaming_metrics_emitter = Mock()
    return broker


@pytest.fixture
def mock_market_data_service():
    """Create mock MarketDataService."""
    service = Mock()
    service._update_mark_window = Mock()
    service._mark_lock = threading.RLock()
    service.update_marks = AsyncMock()
    return service


@pytest.fixture
def mock_risk_manager():
    """Create mock risk manager."""
    risk = Mock()
    risk.last_mark_update = {}
    return risk


@pytest.fixture
def mock_event_store():
    """Create mock event store."""
    store = Mock()
    store.append_metric = Mock()
    return store


@pytest.fixture
def mock_market_monitor():
    """Create mock market monitor."""
    monitor = Mock()
    monitor.record_update = Mock()
    return monitor


@pytest.fixture
def streaming_service(
    mock_broker,
    mock_market_data_service,
    mock_risk_manager,
    mock_event_store,
    mock_market_monitor,
):
    """Create StreamingService instance for testing."""
    service = StreamingService(
        symbols=["BTC-USD", "ETH-USD"],
        broker=mock_broker,
        market_data_service=mock_market_data_service,
        risk_manager=mock_risk_manager,
        event_store=mock_event_store,
        market_monitor=mock_market_monitor,
        bot_id="test_bot",
    )
    yield service
    service.stop()


@pytest.fixture
def metrics_server_mock():
    """Create a mock metrics server with streaming hooks."""
    server = Mock()
    server.set_streaming_context = Mock()
    server.update_streaming_status = Mock()
    server.record_streaming_message = Mock()
    server.record_streaming_reconnect = Mock()
    server.record_streaming_heartbeat = Mock()
    server.update_streaming_fallback = Mock()
    return server


@pytest.fixture
def streaming_service_with_metrics(
    mock_broker,
    mock_market_data_service,
    mock_risk_manager,
    mock_event_store,
    mock_market_monitor,
    metrics_server_mock,
):
    """Create StreamingService instance with metrics integration for testing."""
    service = StreamingService(
        symbols=["BTC-USD", "ETH-USD"],
        broker=mock_broker,
        market_data_service=mock_market_data_service,
        risk_manager=mock_risk_manager,
        event_store=mock_event_store,
        market_monitor=mock_market_monitor,
        bot_id="test_bot",
        metrics_server=metrics_server_mock,
        profile="staging",
        stream_name="test_ws",
        rest_poll_interval=0.05,
    )
    yield service
    service.stop()


class TestStreamingServiceStartStop:
    """Test start/stop semantics and thread lifecycle."""

    def test_start_creates_thread(self, streaming_service, mock_broker):
        """Test that start() creates and starts a daemon thread."""
        # Setup: mock stream to return empty list (immediate exit)
        mock_broker.stream_orderbook.return_value = []

        # Execute
        streaming_service.start(level=1)

        # Verify thread created
        assert streaming_service._ws_thread is not None
        assert streaming_service._ws_thread.daemon is True
        assert streaming_service._ws_stop is not None

        # Wait for thread to complete
        streaming_service._ws_thread.join(timeout=1.0)

    def test_start_with_no_symbols_skips(self, streaming_service):
        """Test that start() with empty symbols list does nothing."""
        streaming_service.symbols = []

        streaming_service.start(level=1)

        assert streaming_service._ws_thread is None

    def test_start_idempotent_if_already_running(self, streaming_service, mock_broker):
        """Test that start() is idempotent if thread already running."""

        # Setup: mock a long-running stream
        def slow_stream():
            yield {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50001"}
            # Keep yielding until stop event set
            import time

            while not streaming_service._ws_stop.is_set():
                time.sleep(0.01)

        mock_broker.stream_orderbook.return_value = slow_stream()

        # Start once
        streaming_service.start(level=1)
        first_thread = streaming_service._ws_thread

        # Try to start again
        streaming_service.start(level=1)

        # Should be same thread
        assert streaming_service._ws_thread is first_thread

        # Cleanup
        streaming_service.stop()

    def test_start_validates_level(self, streaming_service, mock_broker):
        """Test that start() validates and defaults level parameter."""
        mock_broker.stream_orderbook.return_value = []

        # Test None defaults to 1
        streaming_service.start(level=None)
        mock_broker.stream_orderbook.assert_called_with(["BTC-USD", "ETH-USD"], level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Reset
        streaming_service._ws_thread = None
        streaming_service._ws_stop = None

        # Test invalid level defaults to 1
        streaming_service.start(level="invalid")
        assert mock_broker.stream_orderbook.call_args[1]["level"] == 1
        streaming_service._ws_thread.join(timeout=1.0)

    def test_stop_sets_event_and_joins_thread(self, streaming_service, mock_broker):
        """Test that stop() sets stop event and joins thread."""

        def controlled_stream():
            # Yield one message then check stop event
            yield {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50001"}
            import time

            while not streaming_service._ws_stop.is_set():
                time.sleep(0.01)

        mock_broker.stream_orderbook.return_value = controlled_stream()

        streaming_service.start(level=1)
        assert streaming_service.is_running()

        streaming_service.stop()

        # Verify cleanup
        assert streaming_service._ws_thread is None
        assert streaming_service._ws_stop is None

    def test_stop_idempotent_if_not_running(self, streaming_service):
        """Test that stop() is safe to call when not running."""
        # Should not raise
        streaming_service.stop()

        # Verify no thread exists
        assert not hasattr(streaming_service, "_ws_thread") or streaming_service._ws_thread is None

    def test_is_running_reflects_thread_state(self, streaming_service, mock_broker):
        """Test that is_running() accurately reflects thread state."""
        # Initially not running
        assert not streaming_service.is_running()

        # Start streaming
        mock_broker.stream_orderbook.return_value = []
        streaming_service.start(level=1)

        # May be running briefly
        # (can't reliably assert True due to race - thread might finish immediately)

        # After join, definitely not running
        streaming_service._ws_thread.join(timeout=1.0)
        streaming_service._ws_thread = None
        assert not streaming_service.is_running()


class TestStreamingServiceFallback:
    """Test orderbook → trades stream fallback."""

    def test_falls_back_to_trades_on_orderbook_error(
        self, streaming_service, mock_broker, mock_event_store
    ):
        """Test fallback from orderbook to trades stream on exception."""
        # Setup: orderbook raises, trades succeeds
        mock_broker.stream_orderbook.side_effect = RuntimeError("Orderbook not available")
        mock_broker.stream_trades.return_value = [
            {"product_id": "BTC-USD", "price": "50000"},
        ]

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify fallback occurred
        mock_broker.stream_orderbook.assert_called_once()
        mock_broker.stream_trades.assert_called_once_with(["BTC-USD", "ETH-USD"])

    def test_exits_if_both_streams_fail(self, streaming_service, mock_broker, mock_event_store):
        """Test that service exits gracefully if both streams fail."""
        # Setup: both streams raise
        mock_broker.stream_orderbook.side_effect = RuntimeError("Orderbook error")
        mock_broker.stream_trades.side_effect = RuntimeError("Trades error")

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify both attempted
        mock_broker.stream_orderbook.assert_called_once()
        mock_broker.stream_trades.assert_called_once()

        # Should log exit event (even though stream failed to start)
        # Note: exit event logged in finally block, so should still be called
        assert any(
            call_args[0][1].get("event_type") == "ws_stream_exit"
            for call_args in mock_event_store.append_metric.call_args_list
        )


class TestStreamingServiceMarkUpdates:
    """Test mark window updates via MarketDataService."""

    def test_updates_mark_window_from_orderbook(
        self,
        streaming_service,
        mock_broker,
        mock_market_data_service,
        mock_risk_manager,
        mock_event_store,
        mock_market_monitor,
    ):
        """Test that orderbook bid/ask updates mark window."""
        # Setup
        mock_broker.stream_orderbook.return_value = [
            {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50002"},
        ]

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify mark window updated with midpoint
        expected_mark = Decimal("50001")  # (50000 + 50002) / 2
        mock_market_data_service._update_mark_window.assert_called_with("BTC-USD", expected_mark)

    def test_updates_mark_window_from_trades(
        self,
        streaming_service,
        mock_broker,
        mock_market_data_service,
    ):
        """Test that trades stream updates mark window."""
        # Setup: use trades stream (no bid/ask)
        mock_broker.stream_orderbook.return_value = [
            {"product_id": "ETH-USD", "price": "3000"},
        ]

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify mark window updated with price
        mock_market_data_service._update_mark_window.assert_called_with("ETH-USD", Decimal("3000"))

    def test_skips_invalid_marks(
        self,
        streaming_service,
        mock_broker,
        mock_market_data_service,
    ):
        """Test that invalid marks (≤0 or missing) are skipped."""
        # Setup: stream with invalid marks
        mock_broker.stream_orderbook.return_value = [
            {"product_id": "BTC-USD", "best_bid": "0", "best_ask": "0"},  # Zero mark
            {"product_id": "ETH-USD"},  # Missing price
            {"product_id": "SOL-USD", "price": "-100"},  # Negative mark
        ]

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify no updates occurred
        mock_market_data_service._update_mark_window.assert_not_called()

    def test_skips_messages_without_symbol(
        self,
        streaming_service,
        mock_broker,
        mock_market_data_service,
    ):
        """Test that messages without symbol are skipped."""
        # Setup
        mock_broker.stream_orderbook.return_value = [
            {"best_bid": "50000", "best_ask": "50001"},  # No symbol
            {"symbol": "", "price": "3000"},  # Empty symbol
        ]

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify no updates
        mock_market_data_service._update_mark_window.assert_not_called()


class TestStreamingServiceEventStoreWrites:
    """Test event store metric writes."""

    def test_writes_mark_update_metrics(
        self,
        streaming_service,
        mock_broker,
        mock_event_store,
    ):
        """Test that mark updates write to event store."""
        # Setup
        mock_broker.stream_orderbook.return_value = [
            {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50002"},
        ]

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify metric written
        update_calls = [
            c
            for c in mock_event_store.append_metric.call_args_list
            if c[0][1].get("event_type") == "ws_mark_update"
        ]
        assert len(update_calls) == 1
        assert update_calls[0][0][0] == "test_bot"
        assert update_calls[0][0][1]["symbol"] == "BTC-USD"
        assert update_calls[0][0][1]["mark"] == "50001"

    def test_writes_stream_exit_metric(
        self,
        streaming_service,
        mock_broker,
        mock_event_store,
    ):
        """Test that stream exit writes to event store."""
        # Setup
        mock_broker.stream_orderbook.return_value = []

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify exit metric written
        exit_calls = [
            c
            for c in mock_event_store.append_metric.call_args_list
            if c[0][1].get("event_type") == "ws_stream_exit"
        ]
        assert len(exit_calls) == 1

    def test_writes_stream_error_metric_on_exception(
        self,
        streaming_service,
        mock_broker,
        mock_event_store,
    ):
        """Test that stream errors write to event store."""

        # Setup: stream raises exception
        def failing_stream():
            yield {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50001"}
            raise RuntimeError("Stream error")

        mock_broker.stream_orderbook.return_value = failing_stream()

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify error metric written
        error_calls = [
            c
            for c in mock_event_store.append_metric.call_args_list
            if c[0][1].get("event_type") == "ws_stream_error"
        ]
        assert len(error_calls) == 1
        assert "Stream error" in error_calls[0][0][1]["message"]

    def test_handles_event_store_write_errors(
        self,
        streaming_service,
        mock_broker,
        mock_event_store,
    ):
        """Test that event store write errors don't crash stream."""
        # Setup: event store raises on append
        mock_event_store.append_metric.side_effect = RuntimeError("Event store error")
        mock_broker.stream_orderbook.return_value = [
            {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50001"},
        ]

        # Execute - should not raise
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify append was attempted (and raised)
        assert mock_event_store.append_metric.call_count > 0


class TestStreamingServiceIntegration:
    """Integration tests for StreamingService behavior."""

    def test_updates_risk_manager_timestamps(
        self,
        streaming_service,
        mock_broker,
        mock_risk_manager,
    ):
        """Test that risk manager timestamps are updated."""
        # Setup
        mock_broker.stream_orderbook.return_value = [
            {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50001"},
        ]

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify timestamp updated
        assert "BTC-USD" in mock_risk_manager.last_mark_update
        assert mock_risk_manager.last_mark_update["BTC-USD"] is not None

    def test_records_market_monitor_updates(
        self,
        streaming_service,
        mock_broker,
        mock_market_monitor,
    ):
        """Test that market monitor receives update notifications."""
        # Setup
        mock_broker.stream_orderbook.return_value = [
            {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50001"},
            {"product_id": "ETH-USD", "price": "3000"},
        ]

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify monitor notified
        assert mock_market_monitor.record_update.call_count == 2
        mock_market_monitor.record_update.assert_any_call("BTC-USD")
        mock_market_monitor.record_update.assert_any_call("ETH-USD")

    def test_handles_monitoring_errors_gracefully(
        self,
        streaming_service,
        mock_broker,
        mock_market_monitor,
        mock_market_data_service,
    ):
        """Test that monitoring errors don't prevent mark updates."""
        # Setup: market monitor raises
        mock_market_monitor.record_update.side_effect = RuntimeError("Monitor error")
        mock_broker.stream_orderbook.return_value = [
            {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50001"},
        ]

        # Execute - should not raise
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify mark update still occurred
        mock_market_data_service._update_mark_window.assert_called_once()

    def test_processes_multiple_symbols(
        self,
        streaming_service,
        mock_broker,
        mock_market_data_service,
    ):
        """Test that stream processes updates for multiple symbols."""
        # Setup
        mock_broker.stream_orderbook.return_value = [
            {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50001"},
            {"product_id": "ETH-USD", "best_bid": "3000", "best_ask": "3001"},
            {"product_id": "BTC-USD", "best_bid": "50100", "best_ask": "50101"},
        ]

        # Execute
        streaming_service.start(level=1)
        streaming_service._ws_thread.join(timeout=1.0)

        # Verify all updates processed
        assert mock_market_data_service._update_mark_window.call_count == 3
        calls = mock_market_data_service._update_mark_window.call_args_list
        symbols = [c[0][0] for c in calls]
        assert "BTC-USD" in symbols
        assert "ETH-USD" in symbols

    def test_stop_event_terminates_stream(
        self,
        streaming_service,
        mock_broker,
        mock_market_data_service,
    ):
        """Test that setting stop event terminates stream loop."""

        def infinite_stream():
            count = 0
            while True:
                count += 1
                yield {"product_id": "BTC-USD", "best_bid": "50000", "best_ask": "50001"}
                # Check stop event periodically
                if streaming_service._ws_stop and streaming_service._ws_stop.is_set():
                    break
                if count > 100:  # Safety limit
                    break

        mock_broker.stream_orderbook.return_value = infinite_stream()

        # Start streaming
        streaming_service.start(level=1)

        # Let it run briefly
        import time

        time.sleep(0.1)

        # Stop should terminate
        streaming_service.stop()

        # Verify stopped
        assert not streaming_service.is_running()


class TestStreamingServiceMetricsIntegration:
    """Tests for streaming metrics bridging to MetricsServer."""

    def test_metrics_emitter_registered(self, streaming_service_with_metrics, mock_broker) -> None:
        """Broker receives metrics emitter registration on init."""

        mock_broker.set_streaming_metrics_emitter.assert_called_once()

    def test_metrics_events_bridge_to_metrics_server(
        self, streaming_service_with_metrics, metrics_server_mock
    ) -> None:
        """Streaming events map to MetricsServer recording calls."""

        metrics_server_mock.update_streaming_status.reset_mock()
        streaming_service_with_metrics._handle_streaming_metrics_event({"event_type": "ws_connect"})
        metrics_server_mock.update_streaming_status.assert_called_with(
            True, profile="staging", stream="test_ws"
        )

        metrics_server_mock.record_streaming_message.reset_mock()
        streaming_service_with_metrics._handle_streaming_metrics_event(
            {"event_type": "ws_message", "elapsed_since_last": 0.5, "timestamp": 123.0}
        )
        metrics_server_mock.record_streaming_message.assert_called_with(
            0.5, timestamp=123.0, profile="staging", stream="test_ws"
        )

        metrics_server_mock.record_streaming_reconnect.reset_mock()
        streaming_service_with_metrics._handle_streaming_metrics_event(
            {"event_type": "ws_reconnect_attempt", "attempt": 2}
        )
        metrics_server_mock.record_streaming_reconnect.assert_called_with(
            "attempt", attempt=2, profile="staging", stream="test_ws"
        )

        streaming_service_with_metrics._handle_streaming_metrics_event(
            {"event_type": "ws_reconnect_success", "attempt": 2}
        )
        metrics_server_mock.record_streaming_reconnect.assert_called_with(
            "success", attempt=2, profile="staging", stream="test_ws"
        )

    def test_rest_fallback_starts_and_stops_on_events(
        self, streaming_service_with_metrics, metrics_server_mock
    ) -> None:
        """REST fallback toggles based on streaming disconnect/connect events."""

        service = streaming_service_with_metrics
        metrics_server_mock.update_streaming_fallback.reset_mock()

        try:
            service._handle_streaming_metrics_event({"event_type": "ws_disconnect"})
            time.sleep(0.05)
            assert service._rest_fallback_active is True
            metrics_server_mock.update_streaming_fallback.assert_called_with(
                True, profile="staging", stream="test_ws"
            )

            metrics_server_mock.update_streaming_fallback.reset_mock()
            service._handle_streaming_metrics_event(
                {"event_type": "ws_message", "elapsed_since_last": 0.1, "timestamp": time.time()}
            )
            time.sleep(0.05)
            assert service._rest_fallback_active is False
            metrics_server_mock.update_streaming_fallback.assert_called_with(
                False, profile="staging", stream="test_ws"
            )
        finally:
            service._stop_rest_fallback()

    def test_set_rest_poll_interval_validation(self, streaming_service_with_metrics) -> None:
        """Rest poll interval setter enforces positive floats."""

        service = streaming_service_with_metrics
        service.set_rest_poll_interval(0.25)
        assert service._rest_poll_interval == 0.25

        # Invalid values should not change current interval
        previous = service._rest_poll_interval
        service.set_rest_poll_interval(0)
        assert service._rest_poll_interval == previous

    def test_rest_fallback_without_metrics_server(self, streaming_service) -> None:
        """Fallback still works when no metrics server is attached."""

        service = streaming_service
        service.set_rest_poll_interval(0.05)

        try:
            service._handle_streaming_metrics_event({"event_type": "ws_disconnect"})
            time.sleep(0.05)
            assert service._rest_fallback_active is True

            service._handle_streaming_metrics_event(
                {"event_type": "ws_message", "elapsed_since_last": 0.2, "timestamp": time.time()}
            )
            time.sleep(0.05)
            assert service._rest_fallback_active is False
        finally:
            service._stop_rest_fallback()
