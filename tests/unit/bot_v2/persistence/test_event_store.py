"""Tests for EventStore persistence functionality."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bot_v2.persistence.event_store import EventStore


class TestEventStore:
    """Test the EventStore class."""

    def test_event_store_init_default_path(self) -> None:
        """Test EventStore initialization with default path."""
        store = EventStore()

        # Should create a path ending with events.jsonl
        path_str = str(store.path)
        assert path_str.endswith("events.jsonl")
        assert "events" in path_str

    def test_event_store_init_custom_path(self) -> None:
        """Test EventStore initialization with custom path."""
        custom_root = Path("/tmp/test_events")
        store = EventStore(root=custom_root)

        expected_path = custom_root / "events.jsonl"
        assert store.path == expected_path

    @patch("bot_v2.persistence.event_store.utc_now_iso")
    def test_write_adds_timestamp(self, mock_utc_now: Mock) -> None:
        """Test _write method adds timestamp automatically."""
        mock_utc_now.return_value = "2023-01-01T12:00:00Z"

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            store._write({"bot_id": "b", "type": "test", "test": "data"})

            mock_store_instance.append_jsonl.assert_called_once_with(
                {"bot_id": "b", "type": "test", "test": "data", "time": "2023-01-01T12:00:00Z"}
            )

    @patch("bot_v2.persistence.event_store.utc_now_iso")
    def test_write_preserves_existing_timestamp(self, mock_utc_now: Mock) -> None:
        """Test _write method preserves existing timestamp."""
        mock_utc_now.return_value = "2023-01-01T12:00:00Z"

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            store._write(
                {"bot_id": "b", "type": "test", "time": "2023-01-01T10:00:00Z", "test": "data"}
            )

            # Should not override existing timestamp
            mock_store_instance.append_jsonl.assert_called_once_with(
                {"bot_id": "b", "type": "test", "time": "2023-01-01T10:00:00Z", "test": "data"}
            )

    @patch("bot_v2.persistence.event_store.utc_now_iso")
    def test_append_trade(self, mock_utc_now: Mock) -> None:
        """Test append_trade method."""
        mock_utc_now.return_value = "2023-01-01T12:00:00Z"

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            trade_data = {"symbol": "BTC-USD", "side": "buy", "size": 0.1}
            store.append_trade("bot123", trade_data)

            expected_payload = {
                "type": "trade",
                "bot_id": "bot123",
                "symbol": "BTC-USD",
                "side": "buy",
                "size": 0.1,
                "quantity": "0.1",
                "time": "2023-01-01T12:00:00Z",
            }
            mock_store_instance.append_jsonl.assert_called_once_with(expected_payload)

    @patch("bot_v2.persistence.event_store.utc_now_iso")
    def test_append_position(self, mock_utc_now: Mock) -> None:
        """Test append_position method."""
        mock_utc_now.return_value = "2023-01-01T12:00:00Z"

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            position_data = {
                "symbol": "BTC-USD",
                "quantity": 0.5,
                "mark_price": 51000,
                "unrealized_pnl": 100.0,
                "side": "long",
            }
            store.append_position("bot123", position_data)

            expected_payload = {
                "type": "position",
                "bot_id": "bot123",
                "symbol": "BTC-USD",
                "quantity": "0.5",
                "mark_price": "51000",
                "unrealized_pnl": "100.0",
                "side": "long",
                "time": "2023-01-01T12:00:00Z",
            }
            mock_store_instance.append_jsonl.assert_called_once_with(expected_payload)

    @patch("bot_v2.persistence.event_store.utc_now_iso")
    def test_append_metric(self, mock_utc_now: Mock) -> None:
        """Test append_metric method."""
        mock_utc_now.return_value = "2023-01-01T12:00:00Z"

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            metric_data = {
                "event_type": "portfolio_snapshot",
                "portfolio_value": 10000.0,
                "leverage": 2.5,
            }
            store.append_metric("bot123", metric_data)

            expected_payload = {
                "type": "portfolio_snapshot",
                "bot_id": "bot123",
                "event_type": "portfolio_snapshot",
                "portfolio_value": 10000.0,
                "leverage": 2.5,
                "time": "2023-01-01T12:00:00Z",
            }
            mock_store_instance.append_jsonl.assert_called_once_with(expected_payload)

    @patch("bot_v2.persistence.event_store.utc_now_iso")
    def test_append_error_without_context(self, mock_utc_now: Mock) -> None:
        """Test append_error method without context."""
        mock_utc_now.return_value = "2023-01-01T12:00:00Z"

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            store.append_error("bot123", "Connection failed")

            expected_payload = {
                "type": "error",
                "bot_id": "bot123",
                "message": "Connection failed",
                "time": "2023-01-01T12:00:00Z",
            }
            mock_store_instance.append_jsonl.assert_called_once_with(expected_payload)

    @patch("bot_v2.persistence.event_store.utc_now_iso")
    def test_append_error_with_context(self, mock_utc_now: Mock) -> None:
        """Test append_error method with context."""
        mock_utc_now.return_value = "2023-01-01T12:00:00Z"

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            context = {"retry_count": 3, "last_success": "2023-01-01T11:00:00Z"}
            store.append_error("bot123", "Connection failed", context)

            expected_payload = {
                "type": "error",
                "bot_id": "bot123",
                "message": "Connection failed",
                "retry_count": 3,
                "last_success": "2023-01-01T11:00:00Z",
                "time": "2023-01-01T12:00:00Z",
            }
            mock_store_instance.append_jsonl.assert_called_once_with(expected_payload)

    def test_write_requires_bot_id_and_type(self) -> None:
        store = EventStore()
        with pytest.raises(ValueError):
            store._normalize_payload({"type": "trade"})
        with pytest.raises(ValueError):
            store._normalize_payload({"bot_id": "bot123"})

    def test_append_trade_requires_quantity(self, tmp_path: Path) -> None:
        store = EventStore(root=tmp_path)
        with pytest.raises(ValueError):
            store.append_trade("bot123", {"symbol": "BTC-USD", "side": "buy"})

    def test_append_position_requires_mark_price(self, tmp_path: Path) -> None:
        store = EventStore(root=tmp_path)
        with pytest.raises(ValueError):
            store.append_position("bot123", {"symbol": "BTC-USD", "quantity": 1})

    def test_append_metric_requires_event_type(self, tmp_path: Path) -> None:
        store = EventStore(root=tmp_path)
        with pytest.raises(ValueError):
            store.append_metric("bot123", {"value": 1})

    def test_append_error_requires_message(self, tmp_path: Path) -> None:
        store = EventStore(root=tmp_path)
        with pytest.raises(ValueError):
            store.append_error("bot123", "")

    def test_tail_no_filter(self) -> None:
        """Test tail method without type filtering."""
        events = [
            {"bot_id": "bot123", "type": "trade", "symbol": "BTC-USD"},
            {"bot_id": "bot123", "type": "position", "symbol": "ETH-USD"},
            {"bot_id": "bot456", "type": "trade", "symbol": "BTC-USD"},  # Different bot
            {"bot_id": "bot123", "type": "error", "message": "Failed"},
        ]

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.iter_jsonl.return_value = iter(events)
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            result = store.tail("bot123")

            # Should only return events for bot123, in order
            expected = [
                {"bot_id": "bot123", "type": "trade", "symbol": "BTC-USD"},
                {"bot_id": "bot123", "type": "position", "symbol": "ETH-USD"},
                {"bot_id": "bot123", "type": "error", "message": "Failed"},
            ]
            assert result == expected

    def test_tail_with_type_filter(self) -> None:
        """Test tail method with type filtering."""
        events = [
            {"bot_id": "bot123", "type": "trade", "symbol": "BTC-USD"},
            {"bot_id": "bot123", "type": "position", "symbol": "ETH-USD"},
            {"bot_id": "bot123", "type": "trade", "symbol": "ETH-USD"},
            {"bot_id": "bot123", "type": "error", "message": "Failed"},
        ]

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.iter_jsonl.return_value = iter(events)
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            result = store.tail("bot123", types=["trade"])

            # Should only return trade events for bot123
            expected = [
                {"bot_id": "bot123", "type": "trade", "symbol": "BTC-USD"},
                {"bot_id": "bot123", "type": "trade", "symbol": "ETH-USD"},
            ]
            assert result == expected

    def test_tail_with_multiple_type_filter(self) -> None:
        """Test tail method with multiple type filtering."""
        events = [
            {"bot_id": "bot123", "type": "trade", "symbol": "BTC-USD"},
            {"bot_id": "bot123", "type": "position", "symbol": "ETH-USD"},
            {"bot_id": "bot123", "type": "error", "message": "Failed"},
        ]

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.iter_jsonl.return_value = iter(events)
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            result = store.tail("bot123", types=["trade", "error"])

            # Should only return trade and error events for bot123
            expected = [
                {"bot_id": "bot123", "type": "trade", "symbol": "BTC-USD"},
                {"bot_id": "bot123", "type": "error", "message": "Failed"},
            ]
            assert result == expected

    def test_tail_with_limit(self) -> None:
        """Test tail method with limit."""
        events = [
            {"bot_id": "bot123", "type": "trade", "symbol": "BTC-USD"},
            {"bot_id": "bot123", "type": "position", "symbol": "ETH-USD"},
            {"bot_id": "bot123", "type": "trade", "symbol": "ETH-USD"},
            {"bot_id": "bot123", "type": "error", "message": "Failed"},
        ]

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.iter_jsonl.return_value = iter(events)
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            result = store.tail("bot123", limit=2)

            # Should only return last 2 events for bot123
            expected = [
                {"bot_id": "bot123", "type": "trade", "symbol": "ETH-USD"},
                {"bot_id": "bot123", "type": "error", "message": "Failed"},
            ]
            assert result == expected

    def test_tail_handles_invalid_events(self) -> None:
        """Test tail method handles invalid event data."""
        events = [
            {"bot_id": "bot123", "type": "trade", "symbol": "BTC-USD"},
            "invalid_string",  # Invalid event
            None,  # Invalid event
            {"bot_id": "bot123", "type": "position", "symbol": "ETH-USD"},
            123,  # Invalid event
        ]

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.iter_jsonl.return_value = iter(events)
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            result = store.tail("bot123")

            # Should only return valid dict events for bot123
            expected = [
                {"bot_id": "bot123", "type": "trade", "symbol": "BTC-USD"},
                {"bot_id": "bot123", "type": "position", "symbol": "ETH-USD"},
            ]
            assert result == expected

    def test_tail_handles_exception(self) -> None:
        """Test tail method handles exceptions gracefully."""
        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.iter_jsonl.side_effect = Exception("File not found")
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            result = store.tail("bot123")

            # Should return empty list on exception
            assert result == []

    def test_tail_empty_result(self) -> None:
        """Test tail method returns empty list when no matching events."""
        events = [
            {"bot_id": "bot456", "type": "trade", "symbol": "BTC-USD"},  # Different bot
            {"bot_id": "bot789", "type": "position", "symbol": "ETH-USD"},  # Different bot
        ]

        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_store_instance.iter_jsonl.return_value = iter(events)
            mock_json_store.return_value = mock_store_instance

            store = EventStore()
            result = store.tail("bot123")

            # Should return empty list for non-existent bot
            assert result == []

    def test_integration_workflow(self) -> None:
        """Test complete workflow of event storage and retrieval."""
        with patch("bot_v2.persistence.event_store.JsonFileStore") as mock_json_store:
            mock_store_instance = Mock()
            mock_json_store.return_value = mock_store_instance

            store = EventStore()

            # Add different types of events - just verify they call append_jsonl correctly
            store.append_trade(
                "bot123",
                {"symbol": "BTC-USD", "side": "buy", "quantity": 1, "price": "market"},
            )
            store.append_position(
                "bot123",
                {"symbol": "BTC-USD", "quantity": 0.5, "mark_price": 51000},
            )
            store.append_error("bot123", "Test error")
            store.append_trade(
                "bot456", {"symbol": "ETH-USD", "side": "sell", "quantity": 2}
            )  # Different bot

            # Verify all events were written
            assert mock_store_instance.append_jsonl.call_count == 4

            # Check that calls were made with correct structure
            calls = mock_store_instance.append_jsonl.call_args_list

            # First call - trade for bot123
            assert calls[0][0][0]["type"] == "trade"
            assert calls[0][0][0]["bot_id"] == "bot123"
            assert calls[0][0][0]["symbol"] == "BTC-USD"

            # Second call - position for bot123
            assert calls[1][0][0]["type"] == "position"
            assert calls[1][0][0]["bot_id"] == "bot123"

            # Third call - error for bot123
            assert calls[2][0][0]["type"] == "error"
            assert calls[2][0][0]["bot_id"] == "bot123"

            # Fourth call - trade for bot456
            assert calls[3][0][0]["type"] == "trade"
            assert calls[3][0][0]["bot_id"] == "bot456"

            # All calls should have timestamps
            for call in calls:
                assert "time" in call[0][0]
