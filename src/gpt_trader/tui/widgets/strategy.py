from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.staleness_helpers import (
    get_empty_state_config,
    get_freshness_display,
    get_staleness_banner,
)
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.thresholds import (
    get_confidence_label,
    get_confidence_status,
    get_status_color,
)
from gpt_trader.tui.types import StrategyState
from gpt_trader.tui.widgets.tile_states import TileBanner, TileEmptyState
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class StrategyWidget(Static):
    """Displays strategy status and decisions."""

    # Styles moved to styles/widgets/strategy.tcss

    # Reactive state property for automatic updates
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update strategy automatically."""
        if state is None:
            return

        logger.debug(
            f"[StrategyWidget] State update: "
            f"strategies={len(state.strategy_data.active_strategies)}, "
            f"decisions={len(state.strategy_data.last_decisions)}"
        )

        self.update_strategy(state.strategy_data)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="strategy-header"):
            yield Label("STRATEGY DECISIONS", classes="widget-header")
            yield Label("", id="strategy-freshness", classes="data-state-label")
        yield TileBanner(id="strategy-banner", classes="tile-banner hidden")
        table = DataTable(id="strategy-table", zebra_stripes=True)
        table.can_focus = True
        table.cursor_type = "row"
        yield table
        yield TileEmptyState(
            title="No Decisions Yet",
            subtitle="Waiting for strategy signals...",
            icon="◇",
            actions=["[S] Start Bot", "[R] Refresh"],
            id="strategy-empty",
        )

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Action", "Confidence", "Reason", "Time")

        # Register with state registry for state updates
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from state registry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Called by StateRegistry when state changes."""
        # Update staleness banner
        try:
            banner = self.query_one("#strategy-banner", TileBanner)
            staleness_result = get_staleness_banner(state)
            if staleness_result:
                banner.update_banner(staleness_result[0], severity=staleness_result[1])
            else:
                banner.update_banner("")
        except Exception:
            pass

        # Update freshness indicator
        try:
            indicator = self.query_one("#strategy-freshness", Label)
            freshness = get_freshness_display(state)
            if freshness:
                text, css_class = freshness
                indicator.update(text)
                indicator.remove_class("fresh", "stale", "critical")
                indicator.add_class(css_class)
            else:
                indicator.update("")
                indicator.remove_class("fresh", "stale", "critical")
        except Exception:
            pass

        self.state = state

    @safe_update
    def update_strategy(self, data: StrategyState) -> None:
        table = self.query_one(DataTable)
        empty_state = self.query_one("#strategy-empty", TileEmptyState)

        # Handle empty state with actionable hint
        if not data.last_decisions:
            if table.row_count > 0:
                table.clear()
            table.display = False
            empty_state.display = True

            # Use shared empty state config for consistency
            if self.state is not None:
                bot_running = getattr(self.state, "running", False)
                mode = getattr(self.state, "data_source_mode", "demo")
                conn_status = ""
                try:
                    conn_status = str(self.state.system_data.connection_status or "")
                except Exception:
                    pass

                config = get_empty_state_config(
                    data_type="Strategy",
                    bot_running=bot_running,
                    data_source_mode=mode,
                    connection_status=conn_status,
                )
                empty_state.update_state(
                    title=config["title"],
                    subtitle=config["subtitle"],
                    icon=config["icon"],
                    actions=config["actions"],
                )
            return
        else:
            table.display = True
            empty_state.display = False

        # Get current row keys for delta updates
        existing_keys = set(table.rows.keys())
        new_keys = set(data.last_decisions.keys())

        # Remove symbols no longer in decisions
        for key in existing_keys - new_keys:
            try:
                table.remove_row(key)
            except Exception:
                pass

        # Get column keys for cell updates
        columns = list(table.columns.keys()) if table.columns else []

        for symbol, decision in data.last_decisions.items():
            action = decision.action.upper()
            reason = decision.reason

            # Format confidence with badge (e.g., "0.75 HIGH")
            conf_status = get_confidence_status(decision.confidence)
            conf_label = get_confidence_label(conf_status)
            conf_color = get_status_color(conf_status)
            confidence = f"[{conf_color}]{decision.confidence:.2f} {conf_label}[/{conf_color}]"

            # Format timestamp
            time_str = ""
            if decision.timestamp > 0:
                time_str = datetime.fromtimestamp(decision.timestamp).strftime("%H:%M:%S")

            # Color code action
            color = THEME.colors.text_primary
            if action == "BUY":
                color = THEME.colors.success
            elif action == "SELL":
                color = THEME.colors.error
            elif action == "HOLD":
                color = THEME.colors.warning

            formatted_action = f"[{color}]{action}[/{color}]"
            row_data = (symbol, formatted_action, confidence, reason, time_str)

            if symbol in existing_keys:
                # Update existing row in-place
                try:
                    for col_idx, (col_key, value) in enumerate(zip(columns, row_data)):
                        table.update_cell(symbol, col_key, value)
                except Exception:
                    # Fallback: remove and re-add if update fails
                    try:
                        table.remove_row(symbol)
                    except Exception:
                        pass
                    table.add_row(*row_data, key=symbol)
            else:
                # Add new decision
                table.add_row(*row_data, key=symbol)
