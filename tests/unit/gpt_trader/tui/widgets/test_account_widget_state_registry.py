"""Tests for AccountWidget StateRegistry integration."""

import pytest
from textual.app import App, ComposeResult

from gpt_trader.tui.widgets.account import AccountWidget


class TestAccountWidgetStateRegistry:
    """Tests for StateRegistry integration."""

    @pytest.mark.asyncio
    async def test_mount_registers_with_state_registry(self) -> None:
        """Test that widget registers with StateRegistry on mount."""

        class MockStateRegistry:
            def __init__(self):
                self.registered = []

            def register(self, widget):
                self.registered.append(widget)

            def unregister(self, widget):
                if widget in self.registered:
                    self.registered.remove(widget)

        class TestAppWithRegistry(App):
            def __init__(self):
                super().__init__()
                self.state_registry = MockStateRegistry()

            def compose(self) -> ComposeResult:
                yield AccountWidget(id="test-widget")

        app = TestAppWithRegistry()

        async with app.run_test():
            widget = app.query_one(AccountWidget)
            assert widget in app.state_registry.registered
