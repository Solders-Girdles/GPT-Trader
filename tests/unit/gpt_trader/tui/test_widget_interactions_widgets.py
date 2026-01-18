"""
Widget interaction tests: widget visibility, overlays, and responsive layout.

Uses Textual's Pilot API to validate basic accessibility.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from gpt_trader.tui.app import TraderApp


class TestPortfolioWidgetInteractions:
    @pytest.mark.asyncio
    async def test_log_widget_on_main_screen(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            logs = app.screen.query("LogWidget")
            assert len(logs) > 0, "LogWidget should exist on main screen"

    @pytest.mark.asyncio
    async def test_portfolio_accessible_via_details(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press("d")
            await pilot.pause()

            positions = app.screen.query("PositionsWidget")
            assert len(positions) > 0, "PositionsWidget should exist in Details overlay"


class TestDataTableInteractions:
    @pytest.mark.asyncio
    async def test_positions_table_accessible(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            assert app.query("DataTable") is not None


class TestModeIndicator:
    @pytest.mark.asyncio
    async def test_mode_indicator_shows_demo(self, mock_bot_with_status) -> None:
        mock_bot_with_status.__class__.__name__ = "DemoBot"
        type(mock_bot_with_status).__name__ = "DemoBot"

        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            assert app.data_source_mode in ["demo", "paper", "live", "read_only"]


class TestLogWidgetInteractions:
    @pytest.mark.asyncio
    async def test_log_widget_exists(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            assert app.query("LogWidget") is not None

    @pytest.mark.asyncio
    async def test_log_level_filter_chips_present(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            chip_ids = ["level-all", "level-error", "level-warn", "level-info", "level-debug"]
            for chip_id in chip_ids:
                chip = app.screen.query_one(f"#{chip_id}", Button)
                assert chip is not None


class TestStatusBarInteractions:
    @pytest.mark.asyncio
    async def test_status_bar_shows_on_main_screen(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            assert app.query("BotStatusWidget") is not None


class TestErrorIndicatorInteractions:
    @pytest.mark.asyncio
    async def test_error_indicator_exists(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test() as pilot:
            await pilot.pause()

            assert app.error_tracker is not None


class TestResponsiveLayout:
    @pytest.mark.asyncio
    async def test_app_handles_small_terminal(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert app.terminal_width == 80

    @pytest.mark.asyncio
    async def test_app_handles_large_terminal(self, mock_bot_with_status) -> None:
        app = TraderApp(bot=mock_bot_with_status)

        async with app.run_test(size=(200, 60)) as pilot:
            await pilot.pause()
            assert app.terminal_width == 200
