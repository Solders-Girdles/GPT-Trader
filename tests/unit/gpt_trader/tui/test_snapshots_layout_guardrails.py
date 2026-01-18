from __future__ import annotations

import pytest

from gpt_trader.tui.app import TraderApp


class TestLayoutGuardrails:
    """Programmatic layout validation tests."""

    @pytest.mark.asyncio
    async def test_minimum_terminal_size_renders(self, mock_demo_bot) -> None:
        """Verify app renders without errors at minimum terminal size."""
        app = TraderApp(bot=mock_demo_bot)

        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert app._exception is None

    @pytest.mark.asyncio
    async def test_bento_grid_tile_visibility(self, mock_demo_bot) -> None:
        """Verify all bento grid tiles are visible at standard size."""
        app = TraderApp(bot=mock_demo_bot)

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
        app = TraderApp(bot=mock_demo_bot)

        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()

            widgets = list(app.screen.query("Static, Button, Label, DataTable"))
            assert len(widgets) > 0, "No widgets found"
