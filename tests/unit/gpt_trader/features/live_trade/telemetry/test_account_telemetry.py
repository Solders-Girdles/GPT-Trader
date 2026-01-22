"""Tests for AccountTelemetryService initialization and snapshot collection."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

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
