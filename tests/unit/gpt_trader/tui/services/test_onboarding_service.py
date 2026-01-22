"""Tests for onboarding service core functionality."""

import pytest

from gpt_trader.tui.services.onboarding_service import (
    OnboardingService,
    clear_onboarding_service,
    get_onboarding_service,
)


class TestOnboardingService:
    """Tests for OnboardingService."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        clear_onboarding_service()
        yield
        clear_onboarding_service()

    def test_singleton_pattern(self):
        service1 = get_onboarding_service()
        service2 = get_onboarding_service()
        assert service1 is service2

    def test_clear_singleton(self):
        service1 = get_onboarding_service()
        clear_onboarding_service()
        service2 = get_onboarding_service()
        assert service1 is not service2

    def test_get_status_demo_mode_is_ready(self):
        service = OnboardingService()

        class MockState:
            data_source_mode = "demo"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())
        assert status.mode == "demo"
        assert status.is_ready is True

    def test_get_status_paper_mode_needs_credentials(self):
        service = OnboardingService()

        class MockState:
            data_source_mode = "paper"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())
        assert status.mode == "paper"

        creds_item = next((i for i in status.items if i.id == "credentials_valid"), None)
        assert creds_item is not None
        assert creds_item.required is True
        assert creds_item.completed is False

    def test_get_status_returns_four_checklist_items(self):
        service = OnboardingService()

        class MockState:
            data_source_mode = "demo"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())

        assert len(status.items) == 4

        item_ids = [item.id for item in status.items]
        assert "mode_selected" in item_ids
        assert "credentials_valid" in item_ids
        assert "connection_ok" in item_ids
        assert "risk_reviewed" in item_ids

    def test_mode_selected_always_complete(self):
        service = OnboardingService()

        class MockState:
            data_source_mode = "live"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())
        mode_item = next((i for i in status.items if i.id == "mode_selected"), None)
        assert mode_item is not None
        assert mode_item.completed is True

    def test_credentials_skippable_in_demo(self):
        service = OnboardingService()

        class MockState:
            data_source_mode = "demo"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())
        creds_item = next((i for i in status.items if i.id == "credentials_valid"), None)
        assert creds_item is not None
        assert creds_item.skippable is True
        assert creds_item.required is False

    def test_connection_ok_in_demo_mode(self):
        service = OnboardingService()

        class MockState:
            data_source_mode = "demo"
            system_data = type("obj", (object,), {"connection_status": "DISCONNECTED"})()

        status = service.get_status(MockState())
        conn_item = next((i for i in status.items if i.id == "connection_ok"), None)
        assert conn_item is not None
        assert conn_item.completed is True

    def test_connection_checks_status_in_live_mode(self):
        service = OnboardingService()

        class MockStateConnected:
            data_source_mode = "live"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        class MockStateDisconnected:
            data_source_mode = "live"
            system_data = type("obj", (object,), {"connection_status": "DISCONNECTED"})()

        status_connected = service.get_status(MockStateConnected())
        conn_item = next((i for i in status_connected.items if i.id == "connection_ok"), None)
        assert conn_item.completed is True

        status_disconnected = service.get_status(MockStateDisconnected())
        conn_item = next((i for i in status_disconnected.items if i.id == "connection_ok"), None)
        assert conn_item.completed is False

    def test_risk_reviewed_is_optional(self):
        service = OnboardingService()

        class MockState:
            data_source_mode = "demo"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())
        risk_item = next((i for i in status.items if i.id == "risk_reviewed"), None)
        assert risk_item is not None
        assert risk_item.required is False
