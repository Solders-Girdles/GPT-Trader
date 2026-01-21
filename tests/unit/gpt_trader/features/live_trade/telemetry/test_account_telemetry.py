"""Tests for AccountTelemetryService initialization, profile, capabilities, and publishing."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import gpt_trader.features.live_trade.telemetry.account as account_module
from gpt_trader.features.live_trade.telemetry.account import AccountTelemetryService


class TestAccountTelemetryServiceInit:
    """Tests for AccountTelemetryService initialization."""

    def test_init_stores_dependencies(self) -> None:
        """Test initialization stores all dependencies."""
        broker = Mock()
        account_manager = Mock()
        event_store = Mock()
        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=event_store,
            bot_id="test_bot",
            profile="default",
        )
        assert service._broker is broker
        assert service._account_manager is account_manager
        assert service._event_store is event_store
        assert service._bot_id == "test_bot"
        assert service._profile == "default"
        assert service._latest_snapshot == {}


class TestUpdateProfile:
    """Tests for update_profile method."""

    def test_update_profile_changes_profile(self) -> None:
        """Test update_profile changes the profile."""
        service = AccountTelemetryService(
            broker=Mock(),
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )
        service.update_profile("production")
        assert service._profile == "production"


class TestAccountTelemetryProfileEdgeCases:
    """Tests for profile-related edge cases in AccountTelemetryService."""

    def test_multiple_profile_updates(self) -> None:
        """Test multiple profile updates."""
        service = AccountTelemetryService(
            broker=Mock(),
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test",
            profile="initial",
        )
        service.update_profile("profile1")
        assert service._profile == "profile1"
        service.update_profile("profile2")
        assert service._profile == "profile2"


class TestSupportsSnapshots:
    """Tests for supports_snapshots method."""

    def test_supports_snapshots_all_methods_present(self) -> None:
        """Test returns True when broker has all required methods."""
        broker = Mock()
        broker.get_key_permissions = Mock()
        broker.get_fee_schedule = Mock()
        broker.get_account_limits = Mock()
        broker.get_transaction_summary = Mock()
        broker.list_payment_methods = Mock()
        broker.list_portfolios = Mock()
        service = AccountTelemetryService(
            broker=broker,
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )
        assert service.supports_snapshots() is True

    def test_supports_snapshots_missing_method(self) -> None:
        """Test returns False when broker lacks a required method."""
        broker = Mock(spec=["get_key_permissions"])
        service = AccountTelemetryService(
            broker=broker,
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )
        assert service.supports_snapshots() is False

    def test_supports_snapshots_empty_broker(self) -> None:
        """Test returns False when broker has no methods."""
        broker = Mock(spec=[])
        service = AccountTelemetryService(
            broker=broker,
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )
        assert service.supports_snapshots() is False


class TestLatestSnapshot:
    """Tests for latest_snapshot property."""

    def test_latest_snapshot_returns_copy(self) -> None:
        """Test latest_snapshot returns a copy, not the original."""
        service = AccountTelemetryService(
            broker=Mock(),
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )
        service._latest_snapshot = {"key": "value"}
        snapshot = service.latest_snapshot
        assert snapshot == {"key": "value"}
        assert snapshot is not service._latest_snapshot

    def test_latest_snapshot_empty_initially(self) -> None:
        """Test latest_snapshot is empty initially."""
        service = AccountTelemetryService(
            broker=Mock(),
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )
        assert service.latest_snapshot == {}


class TestPublishSnapshot:
    """Tests for _publish_snapshot method."""

    def test_publish_snapshot_emits_metric(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_publish_snapshot_file_write_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
