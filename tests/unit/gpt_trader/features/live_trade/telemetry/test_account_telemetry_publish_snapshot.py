"""Tests for AccountTelemetryService snapshot publishing."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import gpt_trader.features.live_trade.telemetry.account as account_module
from gpt_trader.features.live_trade.telemetry.account import AccountTelemetryService


class TestPublishSnapshot:
    """Tests for _publish_snapshot method."""

    def test_publish_snapshot_emits_metric(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test _publish_snapshot emits metric to event store."""
        mock_emit_metric = Mock()
        monkeypatch.setattr(account_module, "emit_metric", mock_emit_metric)

        mock_runtime_dir = Mock()
        mock_runtime_dir.__truediv__ = Mock(return_value=Mock())
        mock_runtime_dir.__truediv__.return_value.__truediv__ = Mock(return_value=Mock())
        mock_file_path = Mock()
        mock_file_path.parent.mkdir = Mock()
        mock_file_path.open = Mock()
        mock_runtime_dir.__truediv__.return_value.__truediv__.return_value = mock_file_path
        monkeypatch.setattr(account_module, "RUNTIME_DATA_DIR", mock_runtime_dir)

        event_store = Mock()
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

    def test_publish_snapshot_file_write_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test _publish_snapshot handles file write errors gracefully."""
        mock_emit_metric = Mock()
        monkeypatch.setattr(account_module, "emit_metric", mock_emit_metric)

        mock_path = Mock()
        mock_path.parent.mkdir.side_effect = PermissionError("No write access")
        mock_runtime_dir = Mock()
        mock_runtime_dir.__truediv__ = Mock(return_value=Mock())
        mock_runtime_dir.__truediv__.return_value.__truediv__ = Mock(return_value=mock_path)
        monkeypatch.setattr(account_module, "RUNTIME_DATA_DIR", mock_runtime_dir)

        service = AccountTelemetryService(
            broker=Mock(),
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test_bot",
            profile="prod",
        )

        service._publish_snapshot({"balance": "1000"})

        mock_emit_metric.assert_called_once()
