"""
Visual regression snapshots: main screen + theme variants.

Update baselines:
    pytest tests/unit/gpt_trader/tui/test_snapshots_*.py --snapshot-update
"""

from __future__ import annotations

from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.theme import ThemeMode


class TestMainScreenSnapshots:
    """Snapshot tests for the main trading screen at various sizes."""

    def test_main_screen_initial_state(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in its initial stopped state."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
        )

    def test_main_screen_compact_80x24(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in minimum compact terminal (80x24)."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(80, 24),
        )

    def test_main_screen_standard_120x40(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in standard terminal (120x40)."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
        )

    def test_main_screen_wide_200x50(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen in wide terminal (200x50)."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(200, 50),
        )


class TestThemeSnapshots:
    """Snapshot tests for different theme variations."""

    def test_main_screen_dark_theme(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen with dark theme (default)."""

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
        )

    def test_main_screen_light_theme(self, snap_compare, mock_demo_bot):
        """Snapshot test for MainScreen with light theme."""

        async def apply_light_theme(pilot):
            app = pilot.app
            if hasattr(app, "apply_theme_css"):
                app.apply_theme_css(ThemeMode.LIGHT)
            await pilot.pause()

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
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

        def create_app():
            return TraderApp(bot=mock_demo_bot)

        assert snap_compare(
            create_app(),
            terminal_size=(120, 40),
            run_before=apply_high_contrast_theme,
        )
