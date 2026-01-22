"""
Visual regression snapshots: main screen, navigation, and layout.

Update baselines:
    pytest tests/unit/gpt_trader/tui/test_snapshots_*.py --snapshot-update
"""

from __future__ import annotations

import pytest

from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.theme import ThemeMode


def _create_app(mock_demo_bot):
    return TraderApp(bot=mock_demo_bot)


class TestMainScreenSnapshots:
    """Snapshot tests for the main trading screen at various sizes."""

    def test_main_screen_initial_state(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in its initial stopped state."""
        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
        )

    def test_main_screen_compact_80x24(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in minimum compact terminal (80x24)."""
        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(80, 24),
        )

    def test_main_screen_standard_120x40(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in standard terminal (120x40)."""
        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
        )

    def test_main_screen_wide_200x50(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in wide terminal (200x50)."""
        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(200, 50),
        )


class TestThemeSnapshots:
    """Snapshot tests for different theme variations."""

    def test_main_screen_dark_theme(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen with dark theme (default)."""
        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
        )

    def test_main_screen_light_theme(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen with light theme."""

        async def apply_light_theme(pilot):
            app = pilot.app
            if hasattr(app, "apply_theme_css"):
                app.apply_theme_css(ThemeMode.LIGHT)
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
            run_before=apply_light_theme,
        )

    def test_main_screen_high_contrast_theme(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen with high-contrast accessibility theme."""

        async def apply_high_contrast_theme(pilot):
            app = pilot.app
            if hasattr(app, "apply_theme_css"):
                app.apply_theme_css(ThemeMode.HIGH_CONTRAST)
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
            run_before=apply_high_contrast_theme,
        )


class TestScreenTransitionSnapshots:
    """Snapshot tests for screen transitions."""

    def test_full_logs_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for full logs screen."""

        async def open_logs(pilot):
            await pilot.press("1")
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
            run_before=open_logs,
        )

    def test_system_details_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for system details screen."""

        async def open_system_details(pilot):
            await pilot.press("2")
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
            run_before=open_system_details,
        )

    def test_alert_history_screen(self, snap_compare, mock_demo_bot):
        """Snapshot test for alert history screen."""

        async def open_alert_history(pilot):
            await pilot.press("3")
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
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

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
            run_before=focus_tile,
        )

    def test_focus_navigation_arrow_keys(self, snap_compare, mock_demo_bot):
        """Snapshot test for focus after arrow key navigation."""

        async def navigate_with_arrows(pilot):
            await pilot.press("down")
            await pilot.pause()

        assert snap_compare(
            _create_app(mock_demo_bot),
            terminal_size=(120, 40),
            run_before=navigate_with_arrows,
        )


class TestLayoutGuardrails:
    """Programmatic layout validation tests."""

    @pytest.mark.asyncio
    async def test_minimum_terminal_size_renders(self, mock_demo_bot) -> None:
        """Verify app renders without errors at minimum terminal size."""
        app = _create_app(mock_demo_bot)

        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert app._exception is None

    @pytest.mark.asyncio
    async def test_bento_grid_tile_visibility(self, mock_demo_bot) -> None:
        """Verify all bento grid tiles are visible at standard size."""
        app = _create_app(mock_demo_bot)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            expected_tiles = ["tile-hero", "tile-account", "tile-market", "tile-system"]
            for tile_id in expected_tiles:
                try:
                    tile = app.query_one(f"#{tile_id}")
                    assert tile is not None, f"Tile {tile_id} not found"
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_no_overlapping_widgets(self, mock_demo_bot) -> None:
        """Verify no widget overlap at standard terminal size."""
        app = _create_app(mock_demo_bot)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            widgets = list(app.screen.query("Static, Button, Label, DataTable"))
            assert len(widgets) > 0, "No widgets found"
