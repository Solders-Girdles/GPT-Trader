from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Footer, Header, Label

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import (
    AccountWidget,
    BotStatusWidget,
    ContextualFooter,
    LogWidget,
    MarketWatchWidget,
    OrdersWidget,
    PositionsWidget,
    StrategyWidget,
    SystemHealthWidget,
    TradesWidget,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class MainScreen(Screen):
    """Main screen for the GPT-Trader TUI."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield BotStatusWidget(id="bot-status-header")

        # Main Workspace Container - Now horizontal split (30/70)
        with Container(id="main-workspace"):
            # Left Column: Market + Strategy (30% width)
            with Container(id="market-strategy-column"):
                yield MarketWatchWidget(id="dash-market", classes="dashboard-item")
                yield StrategyWidget(id="dash-strategy", classes="dashboard-item")

            # Right Column: Execution + Monitoring (70% width)
            with Container(id="execution-monitoring-column"):
                # Positions (40% of right column = 28% of total screen - LARGEST)
                yield PositionsWidget(id="dash-positions", classes="dashboard-item")

                # Orders (20% of right column)
                yield OrdersWidget(id="dash-orders", classes="dashboard-item")

                # Trades (20% of right column)
                yield TradesWidget(id="dash-trades", classes="dashboard-item")

                # Monitoring Row (20% of right column)
                with Container(id="monitoring-row"):
                    # System + Account (top half, horizontal split)
                    with Container(id="system-account-row"):
                        yield SystemHealthWidget(id="dash-system", classes="dashboard-item")
                        yield AccountWidget(id="dash-account", classes="dashboard-item")

                    # Logs (bottom half, full width, compressed)
                    yield LogWidget(id="dash-logs", classes="dashboard-item compact-logs")

        yield ContextualFooter()

    def on_mount(self) -> None:
        """Called when screen is mounted - trigger initial data load."""
        logger.info("MainScreen mounted, performing initial UI sync")
        # Now that all widgets are mounted, we can safely update them
        # Access the app instance to get state and trigger update
        if hasattr(self.app, "_sync_state_from_bot"):
            self.app._sync_state_from_bot()
            self.update_ui(self.app.tui_state)  # type: ignore[attr-defined]
            if hasattr(self.app, "_pulse_heartbeat"):
                self.app._pulse_heartbeat()
            logger.info("Initial UI sync completed successfully")

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
        except AttributeError as e:
            logger.error(
                f"BotStatusWidget missing expected attribute or state data malformed: {e}",
                exc_info=True,
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data type for bot status update: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to update bot status widget: {e}", exc_info=True)

        # Update Market Data
        try:
            market_widget = self.query_one("#dash-market", MarketWatchWidget)
            market_widget.update_prices(
                state.market_data.prices,
                state.market_data.last_update,
                state.market_data.price_history,
            )
        except AttributeError as e:
            logger.error(
                f"MarketWatchWidget missing expected attribute or market data malformed: {e}",
                exc_info=True,
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data type for market update: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to update market widget: {e}", exc_info=True)

        # Update Strategy
        try:
            strategy_widget = self.query_one("#dash-strategy", StrategyWidget)
            strategy_widget.update_strategy(state.strategy_data)
        except AttributeError as e:
            logger.error(
                f"StrategyWidget missing expected attribute or strategy data malformed: {e}",
                exc_info=True,
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data type for strategy update: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to update strategy widget: {e}", exc_info=True)

        # Update Positions
        try:
            pos_widget = self.query_one("#dash-positions", PositionsWidget)
            pos_widget.update_positions(
                state.position_data.positions,
                state.position_data.total_unrealized_pnl,
                risk_data=state.risk_data.position_leverage,  # Pass risk leverage data
            )
        except AttributeError as e:
            logger.error(
                f"PositionsWidget missing expected attribute or position data malformed: {e}",
                exc_info=True,
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data type for positions update: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to update positions widget: {e}", exc_info=True)

        # Update Orders
        try:
            orders_widget = self.query_one("#dash-orders", OrdersWidget)
            orders_widget.update_orders(state.order_data.orders)
        except AttributeError as e:
            logger.error(
                f"OrdersWidget missing expected attribute or order data malformed: {e}",
                exc_info=True,
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data type for orders update: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to update orders widget: {e}", exc_info=True)

        # Update Trades
        try:
            trades_widget = self.query_one("#dash-trades", TradesWidget)
            trades_widget.update_trades(state.trade_data.trades)
        except AttributeError as e:
            logger.error(
                f"TradesWidget missing expected attribute or trade data malformed: {e}",
                exc_info=True,
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data type for trades update: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to update trades widget: {e}", exc_info=True)

        # Update System Health
        try:
            sys_widget = self.query_one("#dash-system", SystemHealthWidget)
            sys_widget.update_system(state.system_data)
        except AttributeError as e:
            logger.error(
                f"SystemHealthWidget missing expected attribute or system data malformed: {e}",
                exc_info=True,
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data type for system health update: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to update system health widget: {e}", exc_info=True)

        # Update Account
        try:
            acc_widget = self.query_one("#dash-account", AccountWidget)
            acc_widget.update_account(
                state.account_data,
                portfolio_value=state.position_data.equity,
                total_pnl=state.position_data.total_unrealized_pnl,
            )
        except AttributeError as e:
            logger.error(
                f"AccountWidget missing expected attribute or account data malformed: {e}",
                exc_info=True,
            )
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid data type for account update: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to update account widget: {e}", exc_info=True)

        # Logs are updated via the logging handler, not here directly


class FullLogsScreen(Screen):
    """Full-screen log viewer with filtering and expanded view."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("📋 FULL SYSTEM LOGS", classes="header")
        yield LogWidget(id="full-logs", compact_mode=False)  # Expanded mode for full logs screen
        yield Footer()

    def action_dismiss(self) -> None:
        """Close the full logs screen and return to main view."""
        self.app.pop_screen()


class SystemDetailsScreen(Screen):
    """Detailed system health and diagnostics screen."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("⚙️ SYSTEM DETAILS", classes="header")
        with Container(id="system-details-container"):
            yield SystemHealthWidget(
                id="detailed-system", compact_mode=False, classes="dashboard-item"
            )
            yield AccountWidget(id="detailed-account", compact_mode=False, classes="dashboard-item")
        yield Footer()

    def action_dismiss(self) -> None:
        """Close the system details screen and return to main view."""
        self.app.pop_screen()
