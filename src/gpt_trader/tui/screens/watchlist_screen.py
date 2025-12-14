"""
Watchlist Editor Screen for managing tracked symbols.

Provides a modal interface for:
- Viewing current watchlist symbols
- Adding new symbols
- Removing symbols
- Quick-add from current holdings
- Symbol validation (optional)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label, Static

from gpt_trader.tui.services.preferences_service import get_preferences_service
from gpt_trader.tui.theme import THEME
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class WatchlistScreen(ModalScreen):
    """Modal screen for editing the watchlist.

    Features:
    - View and remove current watchlist symbols
    - Add new symbols via text input
    - Quick-add buttons for symbols from current holdings
    - Changes saved on confirm, discarded on cancel

    Keyboard:
    - ESC: Cancel and discard changes
    - Enter: When input focused, add symbol; otherwise save
    - D: Delete selected symbol
    - A: Focus add input
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("d", "delete_selected", "Delete", show=True),
        Binding("a", "focus_input", "Add", show=True),
    ]

    CSS = """
    WatchlistScreen {
        align: center middle;
    }

    #watchlist-modal {
        width: 60;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $bg-primary;
        padding: 1 2;
    }

    #watchlist-title {
        text-style: bold;
        color: $accent;
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }

    #current-symbols {
        height: auto;
        max-height: 15;
        margin-bottom: 1;
    }

    #watchlist-table {
        height: auto;
        max-height: 12;
    }

    .add-section {
        height: auto;
        margin: 1 0;
    }

    .add-row {
        height: auto;
    }

    #symbol-input {
        width: 1fr;
        margin-right: 1;
    }

    #add-btn {
        min-width: 8;
    }

    .holdings-section {
        height: auto;
        margin: 1 0;
    }

    .holdings-label {
        color: $text-muted;
        margin-bottom: 1;
    }

    #holdings-buttons {
        height: auto;
        layout: horizontal;
        overflow-x: auto;
    }

    .holding-btn {
        min-width: 10;
        margin-right: 1;
    }

    .action-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }

    #save-btn {
        margin-right: 1;
    }

    .empty-message {
        color: $text-muted;
        text-align: center;
        padding: 2;
    }
    """

    # Track pending changes
    pending_symbols: reactive[list[str]] = reactive(list, init=False)

    def __init__(self, **kwargs) -> None:
        """Initialize WatchlistScreen."""
        super().__init__(**kwargs)
        self._original_symbols: list[str] = []
        self._holdings: list[str] = []

    def compose(self) -> ComposeResult:
        """Compose the watchlist editor modal."""
        with Vertical(id="watchlist-modal"):
            yield Label("EDIT WATCHLIST", id="watchlist-title")

            # Current symbols section
            with Container(id="current-symbols"):
                yield Label("Current Symbols:", classes="section-label")
                table = DataTable(id="watchlist-table", zebra_stripes=True, cursor_type="row")
                table.can_focus = True
                yield table
                yield Label("No symbols in watchlist", id="empty-message", classes="empty-message")

            # Add symbol section
            with Vertical(classes="add-section"):
                yield Label("Add Symbol:", classes="section-label")
                with Horizontal(classes="add-row"):
                    yield Input(
                        placeholder="Enter symbol (e.g., BTC-USD)",
                        id="symbol-input",
                    )
                    yield Button("Add", id="add-btn", variant="primary")

            # Quick-add from holdings
            with Vertical(classes="holdings-section", id="holdings-section"):
                yield Label("Quick Add from Holdings:", classes="holdings-label")
                yield Horizontal(id="holdings-buttons")

            # Action buttons
            with Horizontal(classes="action-buttons"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn")

    def on_mount(self) -> None:
        """Initialize with current watchlist and holdings."""
        logger.debug("WatchlistScreen mounted")

        # Set up table
        table = self.query_one("#watchlist-table", DataTable)
        table.add_column("#", key="index")
        table.add_column("Symbol", key="symbol")

        # Load current watchlist
        prefs = get_preferences_service()
        self._original_symbols = prefs.get_last_symbols()
        self.pending_symbols = list(self._original_symbols)

        # Load current holdings for quick-add
        self._load_holdings()

        # Initial display
        self._refresh_table()
        self._refresh_holdings_buttons()

    def _load_holdings(self) -> None:
        """Load current holdings for quick-add buttons."""
        try:
            if hasattr(self.app, "tui_state"):
                state: TuiState = self.app.tui_state  # type: ignore[attr-defined]

                # Get symbols from account balances (non-USD assets)
                holdings = set()
                for bal in state.account_data.balances:
                    asset = bal.asset.upper()
                    if asset not in ("USD", "USDC", "USDT"):
                        # Convert to trading pair format
                        holdings.add(f"{asset}-USD")

                # Also include symbols from market data
                for symbol in state.market_data.prices.keys():
                    holdings.add(symbol.upper())

                self._holdings = sorted(holdings)
        except Exception as e:
            logger.debug(f"Failed to load holdings: {e}")
            self._holdings = []

    def _refresh_table(self) -> None:
        """Refresh the symbols table."""
        table = self.query_one("#watchlist-table", DataTable)
        empty_msg = self.query_one("#empty-message", Label)

        table.clear()

        if not self.pending_symbols:
            table.display = False
            empty_msg.display = True
        else:
            table.display = True
            empty_msg.display = False

            for idx, symbol in enumerate(self.pending_symbols, 1):
                table.add_row(str(idx), symbol, key=symbol)

    def _refresh_holdings_buttons(self) -> None:
        """Refresh quick-add buttons from holdings."""
        try:
            container = self.query_one("#holdings-buttons", Horizontal)
            section = self.query_one("#holdings-section", Vertical)

            # Remove existing buttons
            for btn in list(container.query(Button)):
                btn.remove()

            # Filter out symbols already in watchlist
            available = [h for h in self._holdings if h not in self.pending_symbols]

            if not available:
                section.add_class("hidden")
                return

            section.remove_class("hidden")

            # Add buttons for first 5 available
            for symbol in available[:5]:
                btn = Button(symbol, classes="holding-btn", id=f"hold-{symbol}")
                container.mount(btn)

        except Exception as e:
            logger.debug(f"Failed to refresh holdings buttons: {e}")

    def _add_symbol(self, symbol: str) -> bool:
        """Add a symbol to the pending list.

        Args:
            symbol: Symbol to add

        Returns:
            True if added, False if invalid or duplicate.
        """
        symbol = symbol.upper().strip()

        if not symbol:
            self.notify("Please enter a symbol", severity="warning", timeout=2)
            return False

        # Basic validation - must contain a hyphen for pairs
        if "-" not in symbol:
            symbol = f"{symbol}-USD"  # Default to USD pair

        if symbol in self.pending_symbols:
            self.notify(f"{symbol} already in watchlist", severity="warning", timeout=2)
            return False

        self.pending_symbols.append(symbol)
        self._refresh_table()
        self._refresh_holdings_buttons()
        self.notify(f"Added {symbol}", timeout=2)
        return True

    def _remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from the pending list.

        Args:
            symbol: Symbol to remove

        Returns:
            True if removed.
        """
        symbol = symbol.upper().strip()
        if symbol in self.pending_symbols:
            self.pending_symbols.remove(symbol)
            self._refresh_table()
            self._refresh_holdings_buttons()
            self.notify(f"Removed {symbol}", timeout=2)
            return True
        return False

    # === Event Handlers ===

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "add-btn":
            input_widget = self.query_one("#symbol-input", Input)
            symbol = input_widget.value
            if self._add_symbol(symbol):
                input_widget.value = ""
                input_widget.focus()

        elif button_id == "save-btn":
            self._save_and_close()

        elif button_id == "cancel-btn":
            self.action_cancel()

        elif button_id and button_id.startswith("hold-"):
            # Quick-add from holdings
            symbol = button_id[5:]  # Remove "hold-" prefix
            self._add_symbol(symbol)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter in input field."""
        if event.input.id == "symbol-input":
            symbol = event.value
            if self._add_symbol(symbol):
                event.input.value = ""

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection (for potential future drag-reorder)."""
        pass  # Could implement reordering here

    # === Actions ===

    def action_cancel(self) -> None:
        """Cancel and discard changes."""
        self.dismiss(None)

    def action_delete_selected(self) -> None:
        """Delete the selected symbol from the watchlist."""
        try:
            table = self.query_one("#watchlist-table", DataTable)
            cursor = table.cursor_coordinate
            if cursor is None:
                return

            row_key = table.get_row_at(cursor.row)
            if row_key:
                self._remove_symbol(str(row_key))
        except Exception as e:
            logger.debug(f"Delete failed: {e}")

    def action_focus_input(self) -> None:
        """Focus the add symbol input."""
        try:
            self.query_one("#symbol-input", Input).focus()
        except Exception:
            pass

    def _save_and_close(self) -> None:
        """Save changes and close the modal."""
        prefs = get_preferences_service()

        # Only save if changed
        if self.pending_symbols != self._original_symbols:
            prefs.set_last_symbols(self.pending_symbols)
            logger.info(f"Watchlist updated: {len(self.pending_symbols)} symbols")
            self.notify(f"Saved {len(self.pending_symbols)} symbols", timeout=2)

        self.dismiss(self.pending_symbols)
