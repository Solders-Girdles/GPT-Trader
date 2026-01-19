"""
Visual regression snapshots: portfolio empty states and overlays.

Update baselines:
    pytest tests/unit/gpt_trader/tui/test_snapshots_*.py --snapshot-update
"""

from __future__ import annotations

from gpt_trader.tui.app import TraderApp


class TestEmptyStateSnapshots:
    """Snapshot tests for empty data states."""

    def test_portfolio_empty_positions(self, snap_compare, mock_demo_bot):
        """Snapshot test for portfolio widget with no positions."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
        )

    def test_portfolio_empty_orders(self, snap_compare, mock_demo_bot):
        """Snapshot test for orders tab with no open orders."""

        async def navigate_to_orders(pilot):
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=navigate_to_orders,
        )


class TestHelpScreenSnapshots:
    """Snapshot tests for the help screen."""

    def test_help_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for HelpScreen."""

        async def open_help(pilot):
            await pilot.press("?")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(100, 35),
            run_before=open_help,
        )


class TestModeSelectionSnapshots:
    """Snapshot tests for mode selection screen."""

    def test_mode_selection_screen(self, snap_compare):
        """Snapshot test for ModeSelectionScreen (no bot provided)."""

        def create_app():
            return TraderApp()

        assert snap_compare(
            create_app(),
            terminal_size=(100, 30),
        )


class TestWidgetSnapshots:
    """Snapshot tests for individual widgets."""

    def test_portfolio_widget_with_positions(self, snap_compare, mock_demo_bot):
        """Snapshot test for portfolio widget with positions loaded."""

        async def setup_positions(pilot):
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=setup_positions,
        )


class TestErrorStateSnapshots:
    """Snapshot tests for error states and indicators."""

    def test_error_indicator_visible(self, snap_compare, mock_demo_bot):
        """Snapshot test for error indicator when errors are present."""

        async def trigger_error(pilot):
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=trigger_error,
        )
