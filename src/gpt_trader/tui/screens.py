from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Footer, Header, TabbedContent, TabPane

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import (
    AccountWidget,
    BlockChartWidget,
    BotStatusWidget,
    LogWidget,
    MarketWatchWidget,
    OrdersWidget,
    PositionsWidget,
    RiskWidget,
    StrategyWidget,
    SystemHealthWidget,
    TradesWidget,
)


class MainScreen(Screen):
    """Main screen for the GPT-Trader TUI."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield BotStatusWidget(id="bot-status-header")

        with TabbedContent(initial="dashboard"):
            with TabPane("Dashboard", id="dashboard"):
                with Container(id="dashboard-container"):
                    # We reuse widgets but wrap them for the grid
                    yield MarketWatchWidget(id="dash-market", classes="dashboard-item")
                    yield StrategyWidget(id="dash-strategy", classes="dashboard-item")
                    yield PositionsWidget(id="dash-positions", classes="dashboard-item")
                    yield RiskWidget(id="dash-risk", classes="dashboard-item")
                    yield SystemHealthWidget(id="dash-system", classes="dashboard-item")
                    yield LogWidget(id="dash-logs", classes="dashboard-item")

            with TabPane("Market", id="market"):
                yield MarketWatchWidget(id="market-watch-full")
                yield BlockChartWidget(id="chart-full")

            with TabPane("Positions", id="positions"):
                yield PositionsWidget(id="positions-full")
                yield OrdersWidget(id="orders-full")
                yield TradesWidget(id="trades-full")

            with TabPane("Account", id="account"):
                yield AccountWidget(id="account-full")

            with TabPane("System", id="system"):
                yield SystemHealthWidget(id="system-full")
                yield LogWidget(id="logs-full")

        yield Footer()

    def update_ui(self, state: TuiState) -> None:
        """Update widgets from TuiState."""
        # Update Status
        try:
            status_widget = self.query_one(BotStatusWidget)
            status_widget.running = state.running
            status_widget.uptime = state.uptime
            status_widget.equity = state.position_data.equity
        except Exception:
            pass

        # Update Market Data (Dashboard and Full)
        for widget_id in ["#dash-market", "#market-watch-full"]:
            try:
                market_widget = self.query_one(widget_id, MarketWatchWidget)
                market_widget.update_prices(
                    state.market_data.prices,
                    state.market_data.last_update,
                )
            except Exception:
                pass

        # Update Strategy (Dashboard)
        try:
            strategy_widget = self.query_one("#dash-strategy", StrategyWidget)
            strategy_widget.update_strategy(state.strategy_data)
        except Exception:
            pass

        # Update Risk (Dashboard)
        try:
            risk_widget = self.query_one("#dash-risk", RiskWidget)
            risk_widget.update_risk(state.risk_data)
        except Exception:
            pass

        # Update System Health (Dashboard and Full)
        for widget_id in ["#dash-system", "#system-full"]:
            try:
                sys_widget = self.query_one(widget_id, SystemHealthWidget)
                sys_widget.update_system(state.system_data)
            except Exception:
                pass

        # Update Positions (Dashboard and Full)
        for widget_id in ["#dash-positions", "#positions-full"]:
            try:
                pos_widget = self.query_one(widget_id, PositionsWidget)
                pos_widget.update_positions(
                    state.position_data.positions,
                    state.position_data.total_unrealized_pnl,
                )
            except Exception:
                pass

        # Update Chart (Full Market Tab)
        try:
            chart_widget = self.query_one("#chart-full", BlockChartWidget)
            if state.market_data.price_history:
                symbol = next(iter(state.market_data.price_history))
                history = state.market_data.price_history.get(symbol, [])
                chart_widget.update_chart(history)
        except Exception:
            pass

        # Update Orders
        try:
            orders_widget = self.query_one("#orders-full", OrdersWidget)
            orders_widget.update_orders(state.order_data.orders)
        except Exception:
            pass

        # Update Trades
        try:
            trades_widget = self.query_one("#trades-full", TradesWidget)
            trades_widget.update_trades(state.trade_data.trades)
        except Exception:
            pass

        # Update Account
        try:
            account_widget = self.query_one("#account-full", AccountWidget)
            account_widget.update_account(state.account_data)
        except Exception:
            pass
