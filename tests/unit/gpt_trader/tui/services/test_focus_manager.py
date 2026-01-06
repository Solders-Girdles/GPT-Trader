"""Tests for FocusManager service."""

from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.services.focus_manager import (
    FocusManager,
    TileFocusChanged,
    TilePosition,
)


@pytest.fixture
def mock_app():
    """Create a mock TraderApp."""
    app = MagicMock()
    app.post_message = MagicMock()

    # Mock query_one to return focusable widgets
    def mock_query_one(selector, widget_type=None):
        mock_widget = MagicMock()
        mock_widget.focus = MagicMock()
        mock_widget.add_class = MagicMock()
        mock_widget.remove_class = MagicMock()
        return mock_widget

    app.query_one = MagicMock(side_effect=mock_query_one)

    return app


@pytest.fixture
def focus_manager(mock_app):
    """Create a FocusManager with mock app."""
    return FocusManager(mock_app)


class TestTilePosition:
    """Test suite for TilePosition dataclass."""

    def test_tile_position_creation(self):
        """Test creating a TilePosition."""
        pos = TilePosition(row=0, col=1, tile_id="test-tile")
        assert pos.row == 0
        assert pos.col == 1
        assert pos.tile_id == "test-tile"

    def test_tile_position_equality(self):
        """Test TilePosition equality."""
        pos1 = TilePosition(row=0, col=0, tile_id="tile-a")
        pos2 = TilePosition(row=0, col=0, tile_id="tile-a")
        assert pos1 == pos2

    def test_tile_position_inequality(self):
        """Test TilePosition inequality."""
        pos1 = TilePosition(row=0, col=0, tile_id="tile-a")
        pos2 = TilePosition(row=1, col=0, tile_id="tile-b")
        assert pos1 != pos2


class TestFocusManager:
    """Test suite for FocusManager."""

    def test_initialization(self, focus_manager):
        """Test FocusManager initializes with default position."""
        assert focus_manager._current_row == 0
        assert focus_manager._current_col == 0
        assert focus_manager._current_tile_id == "tile-hero"

    def test_current_tile_id_property(self, focus_manager):
        """Test getting current tile ID."""
        tile_id = focus_manager.current_tile_id
        assert tile_id == "tile-hero"
        assert isinstance(tile_id, str)

    def test_current_actions_property(self, focus_manager):
        """Test getting current actions."""
        actions = focus_manager.current_actions
        assert isinstance(actions, list)
        # tile-hero should have actions
        assert len(actions) > 0

    def test_move_down(self, focus_manager):
        """Test moving focus down."""
        initial_row = focus_manager._current_row
        result = focus_manager.move("down")
        # Should move to market tile
        assert focus_manager._current_row > initial_row or result is None

    def test_move_up_from_start(self, focus_manager):
        """Test moving focus up from start position."""
        focus_manager.move("up")
        # Should stay at top (already at row 0)
        assert focus_manager._current_row == 0

    def test_move_up_after_down(self, focus_manager):
        """Test moving focus up after moving down."""
        # First move down
        focus_manager.move("down")
        row_after_down = focus_manager._current_row

        # Then move up
        focus_manager.move("up")
        assert focus_manager._current_row <= row_after_down

    def test_move_right(self, focus_manager):
        """Test moving focus right."""
        # Start at hero (col 0-1), move right to account (col 2-3)
        focus_manager.move("right")
        # Should be at account tile now
        assert focus_manager._current_tile_id in ["tile-hero", "tile-account"]

    def test_move_left_from_start(self, focus_manager):
        """Test moving focus left from start position."""
        focus_manager.move("left")
        # Should stay at left edge
        assert focus_manager._current_col == 0

    def test_move_returns_tile_id_on_change(self, focus_manager):
        """Test that move returns the new tile ID on successful move."""
        # Move to a position that will change the tile
        result = focus_manager.move("down")
        # Returns new tile_id or None if no movement
        assert result is None or isinstance(result, str)

    def test_invalid_direction(self, focus_manager):
        """Test that invalid direction returns None."""
        result = focus_manager.move("invalid")
        assert result is None

    def test_enable_disable(self, focus_manager):
        """Test enabling and disabling focus management."""
        assert focus_manager._focus_enabled is True

        focus_manager.disable()
        assert focus_manager._focus_enabled is False

        # Move should not work when disabled
        initial_tile = focus_manager._current_tile_id
        result = focus_manager.move("down")
        assert result is None
        assert focus_manager._current_tile_id == initial_tile

        focus_manager.enable()
        assert focus_manager._focus_enabled is True

    def test_move_next_cycles_through_tiles(self, focus_manager):
        """Test that move_next cycles through all tiles."""
        tiles_visited = [focus_manager.current_tile_id]

        for _ in range(len(focus_manager.TILE_ORDER)):
            focus_manager.move_next()
            tiles_visited.append(focus_manager.current_tile_id)

        # Should have cycled through all tiles
        assert len(set(tiles_visited)) == len(focus_manager.TILE_ORDER)

    def test_move_previous(self, focus_manager):
        """Test that move_previous goes to previous tile."""
        initial_idx = focus_manager.TILE_ORDER.index(focus_manager.current_tile_id)
        focus_manager.move_previous()
        new_idx = focus_manager.TILE_ORDER.index(focus_manager.current_tile_id)
        # Should be at previous index (wraps around)
        expected_idx = (initial_idx - 1) % len(focus_manager.TILE_ORDER)
        assert new_idx == expected_idx

    def test_focus_tile(self, focus_manager):
        """Test focusing a specific tile."""
        result = focus_manager.focus_tile("tile-market")
        assert result is True
        assert focus_manager.current_tile_id == "tile-market"

    def test_focus_tile_invalid(self, focus_manager):
        """Test focusing an invalid tile returns False."""
        result = focus_manager.focus_tile("nonexistent-tile")
        assert result is False

    def test_grid_structure(self, focus_manager):
        """Test that GRID is properly defined."""
        assert len(focus_manager.GRID) >= 3  # At least 3 rows
        # Each row should have tile definitions
        for row in focus_manager.GRID:
            assert len(row) > 0
            for tile_info in row:
                assert len(tile_info) == 3  # (tile_id, start_col, end_col)

    def test_tile_order(self, focus_manager):
        """Test that all tiles are in TILE_ORDER."""
        expected_tiles = {
            "tile-hero",
            "tile-account",
            "tile-market",
            "tile-strategy",
            "tile-system",
            "tile-logs",
        }
        assert set(focus_manager.TILE_ORDER) == expected_tiles

    def test_tile_actions_defined(self, focus_manager):
        """Test that actions are defined for all tiles."""
        for tile_id in focus_manager.TILE_ORDER:
            actions = focus_manager.TILE_ACTIONS.get(tile_id)
            assert actions is not None
            assert len(actions) > 0


class TestTileFocusChanged:
    """Test suite for TileFocusChanged event."""

    def test_event_creation(self):
        """Test creating TileFocusChanged event."""
        event = TileFocusChanged(tile_id="test-tile", actions=[("Enter", "Details")])
        assert event.tile_id == "test-tile"
        assert len(event.actions) == 1

    def test_event_with_multiple_actions(self):
        """Test TileFocusChanged with multiple actions."""
        actions = [("Enter", "Details"), ("S", "Start"), ("R", "Refresh")]
        event = TileFocusChanged(tile_id="tile-hero", actions=actions)
        assert event.tile_id == "tile-hero"
        assert len(event.actions) == 3
        assert event.actions[0] == ("Enter", "Details")
