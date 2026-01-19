"""Tests for AccountTelemetryService capabilities and snapshot accessors."""

from __future__ import annotations

from unittest.mock import Mock

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
