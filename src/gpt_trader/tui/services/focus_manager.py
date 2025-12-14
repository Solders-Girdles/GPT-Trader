"""
Focus Manager for TUI tile navigation.

Manages 2D grid-based focus navigation between dashboard tiles using arrow keys.
Provides visual focus ring and actions hint coordination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


@dataclass
class TilePosition:
    """Position of a tile in the 2D grid."""

    row: int
    col: int
    tile_id: str


class FocusManager:
    """Manages 2D tile focus navigation.

    Grid layout (matches main_screen.py bento grid):

        Col 0       Col 1       Col 2       Col 3
    Row 0: [  tile-hero (span 2)  ] [tile-account (span 2)]
    Row 1: [  tile-market (span 3)           ] [tile-system]
    Row 2: [          tile-logs (span 4)                   ]
    Row 3: [          (continued)                          ]

    Navigation rules:
    - Arrow keys move spatially based on tile positions
    - Moving right from hero goes to account
    - Moving down from hero/account goes to market/system
    - Moving down from market/system goes to logs
    - Tab cycles linearly through all tiles
    """

    # Grid definition: list of rows, each row is list of (tile_id, start_col, end_col)
    # This accounts for column spans
    GRID = [
        # Row 0: Hero (0-1), Account (2-3)
        [("tile-hero", 0, 1), ("tile-account", 2, 3)],
        # Row 1: Market (0-2), System (3)
        [("tile-market", 0, 2), ("tile-system", 3, 3)],
        # Row 2-3: Logs (0-3)
        [("tile-logs", 0, 3)],
    ]

    # Linear order for Tab navigation
    TILE_ORDER = ["tile-hero", "tile-account", "tile-market", "tile-system", "tile-logs"]

    # Actions available per tile (for hint display)
    TILE_ACTIONS: dict[str, list[tuple[str, str]]] = {
        "tile-hero": [("Enter", "Details"), ("S", "Start/Stop")],
        "tile-account": [("Enter", "Details"), ("C", "Copy")],
        "tile-market": [("Enter", "Details"), ("W", "Watchlist")],
        "tile-system": [("Enter", "Details"), ("R", "Reconnect")],
        "tile-logs": [("Enter", "Full Logs"), ("Space", "Pause")],
    }

    def __init__(self, app: TraderApp) -> None:
        """Initialize FocusManager.

        Args:
            app: Reference to the TraderApp for querying widgets.
        """
        self.app = app
        self._current_row = 0
        self._current_col = 0
        self._current_tile_id = "tile-hero"
        self._focus_enabled = True

    @property
    def current_tile_id(self) -> str:
        """Get the ID of the currently focused tile."""
        return self._current_tile_id

    @property
    def current_actions(self) -> list[tuple[str, str]]:
        """Get actions for the currently focused tile."""
        return self.TILE_ACTIONS.get(self._current_tile_id, [])

    def enable(self) -> None:
        """Enable focus management."""
        self._focus_enabled = True
        self._apply_focus()

    def disable(self) -> None:
        """Disable focus management and clear visual focus."""
        self._focus_enabled = False
        self._clear_all_focus()

    def move(self, direction: str) -> str | None:
        """Move focus in the specified direction.

        Args:
            direction: One of "up", "down", "left", "right"

        Returns:
            The tile_id that received focus, or None if no movement occurred.
        """
        if not self._focus_enabled:
            return None

        new_row = self._current_row
        new_col = self._current_col

        if direction == "up":
            new_row = max(0, self._current_row - 1)
        elif direction == "down":
            new_row = min(len(self.GRID) - 1, self._current_row + 1)
        elif direction == "left":
            new_col = max(0, self._current_col - 1)
        elif direction == "right":
            new_col = min(3, self._current_col + 1)  # Max 4 columns (0-3)
        else:
            logger.warning(f"Invalid focus direction: {direction}")
            return None

        # Find the tile at the new position
        new_tile_id = self._find_tile_at(new_row, new_col)

        if new_tile_id and new_tile_id != self._current_tile_id:
            old_tile_id = self._current_tile_id
            self._current_row = new_row
            self._current_col = new_col
            self._current_tile_id = new_tile_id
            self._apply_focus(old_tile_id)
            logger.debug(f"Focus moved {direction}: {old_tile_id} -> {new_tile_id}")
            return new_tile_id

        return None

    def move_next(self) -> str | None:
        """Move focus to the next tile in linear order (Tab).

        Returns:
            The tile_id that received focus.
        """
        if not self._focus_enabled:
            return None

        current_idx = self.TILE_ORDER.index(self._current_tile_id)
        next_idx = (current_idx + 1) % len(self.TILE_ORDER)
        return self._focus_tile(self.TILE_ORDER[next_idx])

    def move_previous(self) -> str | None:
        """Move focus to the previous tile in linear order (Shift+Tab).

        Returns:
            The tile_id that received focus.
        """
        if not self._focus_enabled:
            return None

        current_idx = self.TILE_ORDER.index(self._current_tile_id)
        prev_idx = (current_idx - 1) % len(self.TILE_ORDER)
        return self._focus_tile(self.TILE_ORDER[prev_idx])

    def focus_tile(self, tile_id: str) -> bool:
        """Focus a specific tile by ID.

        Args:
            tile_id: The ID of the tile to focus.

        Returns:
            True if the tile was focused, False if not found.
        """
        if tile_id not in self.TILE_ORDER:
            return False

        self._focus_tile(tile_id)
        return True

    def _focus_tile(self, tile_id: str) -> str:
        """Internal method to focus a tile and update position.

        Args:
            tile_id: The ID of the tile to focus.

        Returns:
            The tile_id that received focus.
        """
        old_tile_id = self._current_tile_id
        self._current_tile_id = tile_id

        # Update row/col to match the new tile
        for row_idx, row in enumerate(self.GRID):
            for tile_info in row:
                if tile_info[0] == tile_id:
                    self._current_row = row_idx
                    self._current_col = tile_info[1]  # Start column
                    break

        self._apply_focus(old_tile_id)
        return tile_id

    def _find_tile_at(self, row: int, col: int) -> str | None:
        """Find the tile at the given grid position.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)

        Returns:
            The tile_id at that position, or None if out of bounds.
        """
        if row < 0 or row >= len(self.GRID):
            return None

        row_tiles = self.GRID[row]
        for tile_id, start_col, end_col in row_tiles:
            if start_col <= col <= end_col:
                return tile_id

        return None

    def _apply_focus(self, old_tile_id: str | None = None) -> None:
        """Apply visual focus to the current tile.

        Args:
            old_tile_id: The previously focused tile to unfocus.
        """
        try:
            # Remove focus from old tile
            if old_tile_id:
                try:
                    old_tile = self.app.query_one(f"#{old_tile_id}")
                    old_tile.remove_class("tile-focused")
                except Exception:
                    pass

            # Add focus to new tile
            try:
                new_tile = self.app.query_one(f"#{self._current_tile_id}")
                new_tile.add_class("tile-focused")
            except Exception as e:
                logger.debug(f"Could not focus tile {self._current_tile_id}: {e}")

            # Post focus changed event
            self.app.post_message(TileFocusChanged(self._current_tile_id, self.current_actions))

        except Exception as e:
            logger.debug(f"Error applying focus: {e}")

    def _clear_all_focus(self) -> None:
        """Clear focus from all tiles."""
        for tile_id in self.TILE_ORDER:
            try:
                tile = self.app.query_one(f"#{tile_id}")
                tile.remove_class("tile-focused")
            except Exception:
                pass


# Event for focus changes
from textual.message import Message


class TileFocusChanged(Message):
    """Posted when tile focus changes."""

    def __init__(self, tile_id: str, actions: list[tuple[str, str]]) -> None:
        """Initialize event.

        Args:
            tile_id: ID of the newly focused tile.
            actions: List of (key, description) actions for this tile.
        """
        super().__init__()
        self.tile_id = tile_id
        self.actions = actions
