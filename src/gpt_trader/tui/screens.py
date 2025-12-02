from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Footer, Header

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import (
    AccountWidget,
    BotStatusWidget,
    LogWidget,
    MarketWatchWidget,
    OrdersWidget,
    PositionsWidget,
    StrategyWidget,
    SystemHealthWidget,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class MainScreen(Screen):
    """Main screen for the GPT-Trader TUI."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield BotStatusWidget(id="bot-status-header")

        # Main Workspace Container
        with Container(id="main-workspace"):
            # Top Section: Trading & Execution (70% height)
            with Container(id="trading-section"):
                # Left Column: Market & Strategy
                with Container(id="market-column"):
                    yield MarketWatchWidget(id="dash-market", classes="dashboard-item")
                    yield StrategyWidget(id="dash-strategy", classes="dashboard-item")

                # Right Column: Execution & Positions
                with Container(id="execution-column"):
                    yield PositionsWidget(id="dash-positions", classes="dashboard-item")
                    yield OrdersWidget(id="dash-orders", classes="dashboard-item")

            # Bottom Section: Monitoring & Logs (30% height)
            with Container(id="monitoring-section"):
                # Left: System Health
                with Container(id="system-column"):
                    yield SystemHealthWidget(id="dash-system", classes="dashboard-item")
                    yield AccountWidget(id="dash-account", classes="dashboard-item")

                # Right: Logs
                with Container(id="logs-column"):
                    yield LogWidget(id="dash-logs", classes="dashboard-item")

        yield Footer()

    def update_ui(self, state: TuiState) -> None:
        """Update widgets from TuiState."""
        # Update Status
        try:
            status_widget = self.query_one(BotStatusWidget)
            status_widget.running = state.running
            status_widget.uptime = state.uptime
            status_widget.equity = state.position_data.equity
            status_widget.pnl = state.position_data.total_unrealized_pnl
            # Assuming margin usage might be available in account data or calculated
            # For now, we can leave it as default or update if available
        except Exception as e:
            logger.warning(f"Failed to update bot status widget: {e}")

        # Update Market Data
        try:
            market_widget = self.query_one("#dash-market", MarketWatchWidget)
            market_widget.update_prices(
                state.market_data.prices,
                state.market_data.last_update,
                state.market_data.price_history,
            )
        except Exception as e:
            logger.warning(f"Failed to update market widget: {e}")

        # Update Strategy
        try:
            strategy_widget = self.query_one("#dash-strategy", StrategyWidget)
            strategy_widget.update_strategy(state.strategy_data)
        except Exception as e:
            logger.warning(f"Failed to update strategy widget: {e}")

        # Update Positions
        try:
            pos_widget = self.query_one("#dash-positions", PositionsWidget)
            pos_widget.update_positions(
                state.position_data.positions,
                state.position_data.total_unrealized_pnl,
            )
        except Exception as e:
            logger.warning(f"Failed to update positions widget: {e}")

        # Update Orders
        try:
            orders_widget = self.query_one("#dash-orders", OrdersWidget)
            orders_widget.update_orders(state.order_data.orders)
        except Exception as e:
            logger.warning(f"Failed to update orders widget: {e}")

        # Update System Health
        try:
            sys_widget = self.query_one("#dash-system", SystemHealthWidget)
            sys_widget.update_system(state.system_data)
        except Exception as e:
            logger.warning(f"Failed to update system health widget: {e}")

        # Update Account
        try:
            acc_widget = self.query_one("#dash-account", AccountWidget)
            acc_widget.update_account(state.account_data)
        except Exception as e:
            logger.warning(f"Failed to update account widget: {e}")

        # Logs are updated via the logging handler, not here directly
