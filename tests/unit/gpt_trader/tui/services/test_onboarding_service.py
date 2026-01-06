"""Tests for onboarding service functionality."""

import pytest

from gpt_trader.tui.services.onboarding_service import (
    ChecklistItem,
    OnboardingService,
    OnboardingStatus,
    clear_onboarding_service,
    get_onboarding_service,
)


class TestChecklistItem:
    """Tests for ChecklistItem dataclass."""

    def test_default_values(self):
        """Test default values for checklist item."""
        item = ChecklistItem(
            id="test",
            label="Test Item",
            description="Test description",
        )
        assert item.completed is False
        assert item.required is True
        assert item.skippable is False

    def test_completed_item(self):
        """Test completed checklist item."""
        item = ChecklistItem(
            id="test",
            label="Test Item",
            description="Test description",
            completed=True,
        )
        assert item.completed is True

    def test_optional_item(self):
        """Test optional (non-required) checklist item."""
        item = ChecklistItem(
            id="test",
            label="Test Item",
            description="Test description",
            required=False,
        )
        assert item.required is False

    def test_skippable_item(self):
        """Test skippable checklist item."""
        item = ChecklistItem(
            id="test",
            label="Test Item",
            description="Test description",
            skippable=True,
        )
        assert item.skippable is True


class TestOnboardingStatus:
    """Tests for OnboardingStatus dataclass."""

    def test_empty_status_is_ready(self):
        """Test empty status is considered ready."""
        status = OnboardingStatus(items=[], mode="demo")
        assert status.is_ready is True
        assert status.progress_pct == 1.0
        assert status.ready_label == "Ready"

    def test_completed_count(self):
        """Test completed_count property."""
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", completed=True),
                ChecklistItem(id="b", label="B", description="B", completed=False),
                ChecklistItem(id="c", label="C", description="C", completed=True),
            ],
            mode="demo",
        )
        assert status.completed_count == 2

    def test_required_count_excludes_skippable(self):
        """Test required_count excludes skippable items."""
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True),
                ChecklistItem(id="b", label="B", description="B", required=True, skippable=True),
                ChecklistItem(id="c", label="C", description="C", required=False),
            ],
            mode="demo",
        )
        # Only "a" is required and not skippable
        assert status.required_count == 1

    def test_required_completed(self):
        """Test required_completed property."""
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=True, completed=False),
                ChecklistItem(id="c", label="C", description="C", required=False, completed=True),
            ],
            mode="demo",
        )
        # Only required AND completed
        assert status.required_completed == 1

    def test_is_ready_all_required_complete(self):
        """Test is_ready when all required items are complete."""
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=True, completed=True),
                ChecklistItem(id="c", label="C", description="C", required=False, completed=False),
            ],
            mode="paper",
        )
        assert status.is_ready is True

    def test_is_ready_required_incomplete(self):
        """Test is_ready when required items are incomplete."""
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=True, completed=False),
            ],
            mode="paper",
        )
        assert status.is_ready is False

    def test_progress_pct(self):
        """Test progress_pct calculation."""
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", completed=True),
                ChecklistItem(id="b", label="B", description="B", completed=False),
                ChecklistItem(id="c", label="C", description="C", completed=True),
                ChecklistItem(id="d", label="D", description="D", completed=False),
            ],
            mode="demo",
        )
        assert status.progress_pct == 0.5  # 2/4

    def test_ready_label_when_ready(self):
        """Test ready_label when status is ready."""
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
            ],
            mode="paper",
        )
        assert status.ready_label == "Ready"

    def test_ready_label_when_not_ready(self):
        """Test ready_label when status is not ready."""
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=True, completed=False),
                ChecklistItem(id="c", label="C", description="C", required=True, completed=False),
            ],
            mode="paper",
        )
        assert status.ready_label == "1/3"

    def test_get_next_step_returns_first_incomplete_required(self):
        """Test get_next_step returns first incomplete required item."""
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=True, completed=False),
                ChecklistItem(id="c", label="C", description="C", required=True, completed=False),
            ],
            mode="paper",
        )
        next_step = status.get_next_step()
        assert next_step is not None
        assert next_step.id == "b"

    def test_get_next_step_returns_none_when_ready(self):
        """Test get_next_step returns None when all required are complete."""
        status = OnboardingStatus(
            items=[
                ChecklistItem(id="a", label="A", description="A", required=True, completed=True),
                ChecklistItem(id="b", label="B", description="B", required=False, completed=False),
            ],
            mode="paper",
        )
        assert status.get_next_step() is None


class TestOnboardingService:
    """Tests for OnboardingService."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Clear singleton before and after each test."""
        clear_onboarding_service()
        yield
        clear_onboarding_service()

    def test_singleton_pattern(self):
        """Test get_onboarding_service returns same instance."""
        service1 = get_onboarding_service()
        service2 = get_onboarding_service()
        assert service1 is service2

    def test_clear_singleton(self):
        """Test clear_onboarding_service creates new instance."""
        service1 = get_onboarding_service()
        clear_onboarding_service()
        service2 = get_onboarding_service()
        assert service1 is not service2

    def test_get_status_demo_mode_is_ready(self):
        """Test demo mode is always ready."""
        service = OnboardingService()

        # Mock state with demo mode
        class MockState:
            data_source_mode = "demo"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())
        assert status.mode == "demo"
        assert status.is_ready is True

    def test_get_status_paper_mode_needs_credentials(self):
        """Test paper mode requires credentials."""
        service = OnboardingService()

        # Mock state with paper mode but no credentials validated
        class MockState:
            data_source_mode = "paper"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())
        assert status.mode == "paper"

        # Credentials should be required but not complete (no validation cached)
        creds_item = next((i for i in status.items if i.id == "credentials_valid"), None)
        assert creds_item is not None
        assert creds_item.required is True
        # Without validation cache, should be incomplete
        assert creds_item.completed is False

    def test_get_status_returns_four_checklist_items(self):
        """Test status returns expected checklist items."""
        service = OnboardingService()

        class MockState:
            data_source_mode = "demo"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())

        # Should have 4 standard items
        assert len(status.items) == 4

        item_ids = [item.id for item in status.items]
        assert "mode_selected" in item_ids
        assert "credentials_valid" in item_ids
        assert "connection_ok" in item_ids
        assert "risk_reviewed" in item_ids

    def test_mode_selected_always_complete(self):
        """Test mode_selected is always complete (we're running)."""
        service = OnboardingService()

        class MockState:
            data_source_mode = "live"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())
        mode_item = next((i for i in status.items if i.id == "mode_selected"), None)
        assert mode_item is not None
        assert mode_item.completed is True

    def test_credentials_skippable_in_demo(self):
        """Test credentials are skippable in demo mode."""
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
        """Test connection is OK in demo mode."""
        service = OnboardingService()

        class MockState:
            data_source_mode = "demo"
            system_data = type("obj", (object,), {"connection_status": "DISCONNECTED"})()

        status = service.get_status(MockState())
        conn_item = next((i for i in status.items if i.id == "connection_ok"), None)
        assert conn_item is not None
        # Demo mode connection is always OK
        assert conn_item.completed is True

    def test_connection_checks_status_in_live_mode(self):
        """Test connection status is checked in live mode."""
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
        """Test risk_reviewed is optional (not blocking)."""
        service = OnboardingService()

        class MockState:
            data_source_mode = "demo"
            system_data = type("obj", (object,), {"connection_status": "CONNECTED"})()

        status = service.get_status(MockState())
        risk_item = next((i for i in status.items if i.id == "risk_reviewed"), None)
        assert risk_item is not None
        assert risk_item.required is False
