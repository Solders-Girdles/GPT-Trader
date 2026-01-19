"""Tests for AccountTelemetryService snapshot publishing."""

from __future__ import annotations

from unittest.mock import Mock, patch

from gpt_trader.features.live_trade.telemetry.account import AccountTelemetryService


class TestPublishSnapshot:
    """Tests for _publish_snapshot method."""

    @patch("gpt_trader.features.live_trade.telemetry.account.emit_metric")
    @patch("gpt_trader.features.live_trade.telemetry.account.RUNTIME_DATA_DIR")
    def test_publish_snapshot_emits_metric(
        self, mock_runtime_dir: Mock, mock_emit_metric: Mock
    ) -> None:
        """Test _publish_snapshot emits metric to event store."""
        event_store = Mock()
        mock_runtime_dir.__truediv__ = Mock(return_value=Mock())
        mock_runtime_dir.__truediv__.return_value.__truediv__ = Mock(return_value=Mock())
        mock_file_path = Mock()
        mock_file_path.parent.mkdir = Mock()
        mock_file_path.open = Mock()
        mock_runtime_dir.__truediv__.return_value.__truediv__.return_value = mock_file_path

        service = AccountTelemetryService(
            broker=Mock(),
            account_manager=Mock(),
            event_store=event_store,
            bot_id="test_bot",
            profile="prod",
        )

        snapshot = {"balance": "1000", "timestamp": "2024-01-15T12:00:00Z"}

        service._publish_snapshot(snapshot)

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args
        assert call_args[0][0] is event_store
        assert call_args[0][1] == "test_bot"
        assert call_args[0][2]["event_type"] == "account_snapshot"
        assert call_args[0][2]["balance"] == "1000"

    @patch("gpt_trader.features.live_trade.telemetry.account.emit_metric")
    @patch("gpt_trader.features.live_trade.telemetry.account.RUNTIME_DATA_DIR")
    def test_publish_snapshot_file_write_error(
        self, mock_runtime_dir: Mock, mock_emit_metric: Mock
    ) -> None:
        """Test _publish_snapshot handles file write errors gracefully."""
        mock_path = Mock()
        mock_path.parent.mkdir.side_effect = PermissionError("No write access")
        mock_runtime_dir.__truediv__ = Mock(return_value=Mock())
        mock_runtime_dir.__truediv__.return_value.__truediv__ = Mock(return_value=mock_path)

        service = AccountTelemetryService(
            broker=Mock(),
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test_bot",
            profile="prod",
        )

        service._publish_snapshot({"balance": "1000"})

        mock_emit_metric.assert_called_once()
