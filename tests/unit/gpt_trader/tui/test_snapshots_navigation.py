"""
Visual regression snapshots: screen transitions and focus/navigation.

Update baselines:
    pytest tests/unit/gpt_trader/tui/test_snapshots_*.py --snapshot-update
"""

from __future__ import annotations

from gpt_trader.tui.app import TraderApp


class TestScreenTransitionSnapshots:
    """Snapshot tests for screen transitions."""

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

    def test_alert_history_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for alert history screen."""

        async def open_alert_history(pilot):
            await pilot.press("3")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=open_alert_history,
        )


class TestFocusStateSnapshots:
    """Snapshot tests for focus states and keyboard navigation."""

    def test_tile_focus_ring(self, snap_compare, mock_demo_bot):
        """Snapshot test for tile focus ring visibility."""

        async def focus_tile(pilot):
            await pilot.press("tab")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=focus_tile,
        )

    def test_focus_navigation_arrow_keys(self, snap_compare, mock_demo_bot):
        """Snapshot test for focus after arrow key navigation."""

        async def navigate_with_arrows(pilot):
            await pilot.press("down")
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=navigate_with_arrows,
        )
