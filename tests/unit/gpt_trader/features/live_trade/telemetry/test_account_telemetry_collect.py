"""Tests for AccountTelemetryService snapshot collection."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

from gpt_trader.features.live_trade.telemetry.account import AccountTelemetryService


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

        snapshot = service.collect_snapshot()

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


class TestAccountTelemetrySnapshotEdgeCases:
    """Tests for snapshot-related edge cases in AccountTelemetryService."""

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

        snapshot = service.collect_snapshot()

        assert "timestamp" in snapshot
        assert snapshot.get("server_time") is None
