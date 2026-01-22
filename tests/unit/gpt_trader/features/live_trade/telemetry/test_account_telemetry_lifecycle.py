"""Tests for AccountTelemetryService capabilities, publishing, and async lifecycle."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

import gpt_trader.features.live_trade.telemetry.account as account_module
from gpt_trader.features.live_trade.telemetry.account import AccountTelemetryService


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


class TestRunAsync:
    @pytest.mark.asyncio
    async def test_run_exits_when_snapshots_not_supported(self) -> None:
        """Test run exits immediately when broker doesn't support snapshots."""
        broker = Mock(spec=[])  # No required methods

        service = AccountTelemetryService(
            broker=broker,
            account_manager=Mock(),
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )

        result = await asyncio.wait_for(service.run(interval_seconds=1), timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_run_collects_and_publishes(self) -> None:
        """Test run collects snapshot and publishes."""
        broker = Mock()
        broker.get_key_permissions = Mock()
        broker.get_fee_schedule = Mock()
        broker.get_account_limits = Mock()
        broker.get_transaction_summary = Mock()
        broker.list_payment_methods = Mock()
        broker.list_portfolios = Mock()
        broker.get_server_time.return_value = None

        account_manager = Mock()
        account_manager.snapshot.return_value = {"balance": "1000"}

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )

        published = asyncio.Event()

        def _publish_snapshot(snapshot: dict) -> None:
            published.set()

        service._publish_snapshot = Mock(side_effect=_publish_snapshot)

        task = asyncio.create_task(service.run(interval_seconds=10))
        await asyncio.wait_for(published.wait(), timeout=0.2)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        service._publish_snapshot.assert_called()

    @pytest.mark.asyncio
    async def test_run_handles_collection_error(self) -> None:
        """Test run handles errors during collection gracefully."""
        broker = Mock()
        broker.get_key_permissions = Mock()
        broker.get_fee_schedule = Mock()
        broker.get_account_limits = Mock()
        broker.get_transaction_summary = Mock()
        broker.list_payment_methods = Mock()
        broker.list_portfolios = Mock()

        account_manager = Mock()
        account_manager.snapshot.side_effect = RuntimeError("Collection error")

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )
        published = asyncio.Event()

        def _publish_snapshot(snapshot: dict) -> None:
            published.set()

        service._publish_snapshot = Mock(side_effect=_publish_snapshot)

        task = asyncio.create_task(service.run(interval_seconds=10))
        await asyncio.wait_for(published.wait(), timeout=0.2)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        account_manager.snapshot.assert_called()
        service._publish_snapshot.assert_called()
