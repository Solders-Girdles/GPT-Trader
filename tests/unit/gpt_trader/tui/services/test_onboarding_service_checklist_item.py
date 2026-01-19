"""Tests for ChecklistItem onboarding state."""

from gpt_trader.tui.services.onboarding_service import ChecklistItem


class TestChecklistItem:
    """Tests for ChecklistItem dataclass."""

    def test_default_values(self):
        item = ChecklistItem(
            id="test",
            label="Test Item",
            description="Test description",
        )
        assert item.completed is False
        assert item.required is True
        assert item.skippable is False

    def test_completed_item(self):
        item = ChecklistItem(
            id="test",
            label="Test Item",
            description="Test description",
            completed=True,
        )
        assert item.completed is True

    def test_optional_item(self):
        item = ChecklistItem(
            id="test",
            label="Test Item",
            description="Test description",
            required=False,
        )
        assert item.required is False

    def test_skippable_item(self):
        item = ChecklistItem(
            id="test",
            label="Test Item",
            description="Test description",
            skippable=True,
        )
        assert item.skippable is True
