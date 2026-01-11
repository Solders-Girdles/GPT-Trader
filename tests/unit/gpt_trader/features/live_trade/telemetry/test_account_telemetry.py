"""Tests for AccountTelemetryService - account snapshot collection."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from gpt_trader.features.live_trade.telemetry.account import AccountTelemetryService

# ============================================================
# Test: AccountTelemetryService initialization
# ============================================================


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


# ============================================================
# Test: supports_snapshots method
# ============================================================


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
        broker = Mock(spec=["get_key_permissions"])  # Only has one method

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


# ============================================================
# Test: latest_snapshot property
# ============================================================


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


# ============================================================
# Test: update_profile method
# ============================================================


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


# ============================================================
# Test: collect_snapshot method
# ============================================================


class TestCollectSnapshot:
    """Tests for collect_snapshot method."""

    def test_collect_snapshot_success(self) -> None:
        """Test successful snapshot collection."""
        broker = Mock()
        broker.get_server_time.return_value = datetime(2024, 1, 15, 12, 0, 0)

        account_manager = Mock()
        account_manager.snapshot.return_value = {"balance": "1000", "positions": []}

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )

        snapshot = service.collect_snapshot()

        assert snapshot["balance"] == "1000"
        assert snapshot["positions"] == []
        assert "server_time" in snapshot
        assert "timestamp" in snapshot

    def test_collect_snapshot_account_manager_error(self) -> None:
        """Test snapshot collection when account_manager fails."""
        broker = Mock()
        broker.get_server_time.return_value = datetime(2024, 1, 15, 12, 0, 0)

        account_manager = Mock()
        account_manager.snapshot.side_effect = RuntimeError("Manager error")

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )

        # Should not raise
        snapshot = service.collect_snapshot()

        # Should still have server_time and timestamp
        assert "server_time" in snapshot
        assert "timestamp" in snapshot

    def test_collect_snapshot_server_time_error(self) -> None:
        """Test snapshot collection when get_server_time fails."""
        broker = Mock()
        broker.get_server_time.side_effect = RuntimeError("Server error")

        account_manager = Mock()
        account_manager.snapshot.return_value = {"balance": "1000"}

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )

        # Should not raise
        snapshot = service.collect_snapshot()

        assert snapshot["balance"] == "1000"
        assert snapshot["server_time"] is None
        assert "timestamp" in snapshot

    def test_collect_snapshot_updates_latest(self) -> None:
        """Test that collect_snapshot updates _latest_snapshot."""
        broker = Mock()
        broker.get_server_time.return_value = None

        account_manager = Mock()
        account_manager.snapshot.return_value = {"balance": "2000"}

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )

        snapshot = service.collect_snapshot()

        assert service._latest_snapshot == snapshot
        assert service._latest_snapshot["balance"] == "2000"

    def test_collect_snapshot_server_time_none(self) -> None:
        """Test snapshot when server_time returns None."""
        broker = Mock()
        broker.get_server_time.return_value = None

        account_manager = Mock()
        account_manager.snapshot.return_value = {}

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )

        snapshot = service.collect_snapshot()

        assert snapshot["server_time"] is None


# ============================================================
# Test: _publish_snapshot method
# ============================================================


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
        # Make mkdir raise an error
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

        # Should not raise
        service._publish_snapshot({"balance": "1000"})

        # Metric should still be emitted
        mock_emit_metric.assert_called_once()


# ============================================================
# Test: run async method
# ============================================================


class TestRunAsync:
    """Tests for run async method."""

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

        # Should return immediately without looping
        await asyncio.wait_for(service.run(interval_seconds=1), timeout=0.1)

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

        # Run briefly then cancel after first publish
        task = asyncio.create_task(service.run(interval_seconds=10))
        await asyncio.wait_for(published.wait(), timeout=0.2)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have published at least once
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

        # Run briefly then cancel
        task = asyncio.create_task(service.run(interval_seconds=10))
        await asyncio.wait_for(published.wait(), timeout=0.2)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        account_manager.snapshot.assert_called()
        service._publish_snapshot.assert_called()


# ============================================================
# Test: Edge cases
# ============================================================


class TestAccountTelemetryEdgeCases:
    """Tests for edge cases in AccountTelemetryService."""

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

    def test_snapshot_preserves_account_manager_data(self) -> None:
        """Test that snapshot preserves all account manager data."""
        broker = Mock()
        broker.get_server_time.return_value = None

        account_manager = Mock()
        account_manager.snapshot.return_value = {
            "balance": "5000",
            "positions": [{"symbol": "BTC-PERP", "size": "0.5"}],
            "equity": "4500",
            "margin_used": "1000",
        }

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )

        snapshot = service.collect_snapshot()

        assert snapshot["balance"] == "5000"
        assert snapshot["positions"] == [{"symbol": "BTC-PERP", "size": "0.5"}]
        assert snapshot["equity"] == "4500"
        assert snapshot["margin_used"] == "1000"

    def test_collect_snapshot_both_errors(self) -> None:
        """Test collect_snapshot when both account_manager and server_time fail."""
        broker = Mock()
        broker.get_server_time.side_effect = RuntimeError("Server error")

        account_manager = Mock()
        account_manager.snapshot.side_effect = RuntimeError("Manager error")

        service = AccountTelemetryService(
            broker=broker,
            account_manager=account_manager,
            event_store=Mock(),
            bot_id="test",
            profile="default",
        )

        # Should not raise
        snapshot = service.collect_snapshot()

        # Should still have timestamp
        assert "timestamp" in snapshot
        assert snapshot.get("server_time") is None
