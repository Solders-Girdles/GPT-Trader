import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label

from gpt_trader.tui.types import SystemStatus
from gpt_trader.tui.widgets.system import SystemHealthWidget


class SystemHealthTestApp(App):
    def compose(self) -> ComposeResult:
        yield SystemHealthWidget(compact_mode=False)


class TestSystemHealthWidget:
    @pytest.mark.asyncio
    async def test_update_system(self) -> None:
        app = SystemHealthTestApp()

        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)

            # Initial state
            assert widget.system_data.connection_status == "UNKNOWN"

            # Update with new data
            new_data = SystemStatus(
                api_latency=123.45,
                connection_status="CONNECTED",
                rate_limit_usage="15%",
                memory_usage="42MB",
                cpu_usage="5%",
            )
            widget.update_system(new_data)

            # Verify internal state
            assert widget.system_data == new_data

            # Verify UI updates (use str() to get label content in Textual 7.0+)
            assert str(app.query_one("#connection-status", Label).render()) == "CONNECTED"
            assert app.query_one("#connection-status", Label).has_class("status-connected")
            assert str(app.query_one("#latency", Label).render()) == "123ms"
            assert (
                str(app.query_one("#rate-limit", Label).render()) == "15%"
            )  # No prefix in non-compact mode
            assert str(app.query_one("#memory", Label).render()) == "42MB"
            assert str(app.query_one("#cpu", Label).render()) == "5%"

    @pytest.mark.asyncio
    async def test_disconnected_status(self) -> None:
        app = SystemHealthTestApp()
        async with app.run_test():
            widget = app.query_one(SystemHealthWidget)
            new_data = SystemStatus(connection_status="DISCONNECTED")
            widget.update_system(new_data)

            assert str(app.query_one("#connection-status", Label).render()) == "DISCONNECTED"
            assert app.query_one("#connection-status", Label).has_class("status-disconnected")
            assert not app.query_one("#connection-status", Label).has_class("status-connected")
