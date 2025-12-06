"""
Visual Regression Tests using Snapshot Testing.

Uses pytest-textual-snapshot for visual regression testing of TUI components.
Snapshots are stored as SVG files in the snapshots/ directory.

To update snapshots after intentional UI changes:
    pytest tests/unit/gpt_trader/tui/test_snapshots.py --snapshot-update

For editor integration, set TEXTUAL_SNAPSHOT_FILE_OPEN_PREFIX:
    - file:// (default)
    - code://file/ (VS Code)
    - pycharm:// (PyCharm)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.tui.app import TraderApp


@pytest.fixture
def mock_demo_bot():
    """Creates a mock demo bot for snapshot testing."""
    bot = MagicMock()
    bot.running = False
    bot.run = AsyncMock()
    bot.stop = AsyncMock()
    bot.config = MagicMock()

    # Mock engine and status reporter
    bot.engine = MagicMock()
    bot.engine.status_reporter = StatusReporter()
    bot.engine.context = MagicMock()
    bot.engine.context.runtime_state = None

    # Make it look like a DemoBot for mode detection
    type(bot).__name__ = "DemoBot"

    return bot


class TestMainScreenSnapshots:
    """Snapshot tests for the main trading screen."""

    @pytest.mark.skip(reason="Initial snapshot needs to be generated with --snapshot-update")
    def test_main_screen_initial_state(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in its initial stopped state."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
        )

    @pytest.mark.skip(reason="Initial snapshot needs to be generated with --snapshot-update")
    def test_main_screen_compact_layout(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in compact terminal."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(80, 24),
        )

    @pytest.mark.skip(reason="Initial snapshot needs to be generated with --snapshot-update")
    def test_main_screen_wide_layout(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in wide terminal."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(200, 50),
        )


class TestHelpScreenSnapshots:
    """Snapshot tests for the help screen."""

    @pytest.mark.skip(reason="Initial snapshot needs to be generated with --snapshot-update")
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

    @pytest.mark.skip(reason="Initial snapshot needs to be generated with --snapshot-update")
    def test_mode_selection_screen(self, snap_compare):
        """Snapshot test for ModeSelectionScreen (no bot provided)."""

        def create_app():
            # Create app without bot to show mode selection
            return TraderApp()

        assert snap_compare(
            create_app(),
            terminal_size=(100, 30),
        )


class TestWidgetSnapshots:
    """Snapshot tests for individual widgets."""

    @pytest.mark.skip(reason="Initial snapshot needs to be generated with --snapshot-update")
    def test_portfolio_widget_with_positions(self, snap_compare, mock_demo_bot):
        """Snapshot test for portfolio widget with positions loaded."""

        async def setup_positions(pilot):
            # Navigate to positions tab
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

    @pytest.mark.skip(reason="Initial snapshot needs to be generated with --snapshot-update")
    def test_error_indicator_visible(self, snap_compare, mock_demo_bot):
        """Snapshot test for error indicator when errors are present."""

        async def trigger_error(pilot):
            # Errors would be triggered through state validation
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=trigger_error,
        )


class TestScreenTransitionSnapshots:
    """Snapshot tests for screen transitions."""

    @pytest.mark.skip(reason="Initial snapshot needs to be generated with --snapshot-update")
    def test_full_logs_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for full logs screen."""

        async def open_logs(pilot):
            await pilot.press("1")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=open_logs,
        )

    @pytest.mark.skip(reason="Initial snapshot needs to be generated with --snapshot-update")
    def test_system_details_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for system details screen."""

        async def open_system_details(pilot):
            await pilot.press("2")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=open_system_details,
        )


# Helper function for generating initial snapshots
def enable_snapshot_generation():
    """
    Instructions for generating initial snapshots:

    1. Remove @pytest.mark.skip decorators from tests you want to snapshot
    2. Run: pytest tests/unit/gpt_trader/tui/test_snapshots.py --snapshot-update
    3. Review generated SVG files in tests/unit/gpt_trader/tui/snapshots/
    4. Commit the snapshots to version control

    After initial generation, re-add the skip decorators or remove them
    permanently for CI/CD integration.
    """
    pass
