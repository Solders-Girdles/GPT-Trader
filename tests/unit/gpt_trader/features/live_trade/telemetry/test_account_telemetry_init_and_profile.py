"""Tests for AccountTelemetryService initialization and profile updates."""

from __future__ import annotations

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
