"""
TUI State Management.
"""

from __future__ import annotations

from typing import Any

from textual.reactive import reactive
from textual.widget import Widget

from gpt_trader.tui.types import (
    AccountBalance,
    AccountSummary,
    ActiveOrders,
    DecisionData,
    MarketState,
    Order,
    PortfolioSummary,
    Position,
    RiskState,
    StrategyState,
    SystemStatus,
    Trade,
    TradeHistory,
)


class TuiState(Widget):
    """
    Reactive state for the TUI.
    Acts as a ViewModel that the App observes.
    """

    # Reactive properties that widgets can watch
    running = reactive(False)
    uptime = reactive(0.0)
    cycle_count = reactive(0)

    # We use reactive for complex objects too, but need to be careful about mutation
    # For simple updates, replacing the whole object triggers the watcher
    market_data = reactive(MarketState())
    position_data = reactive(PortfolioSummary())
    order_data = reactive(ActiveOrders())
    trade_data = reactive(TradeHistory())
    account_data = reactive(AccountSummary())
    strategy_data = reactive(StrategyState())
    risk_data = reactive(RiskState())
    system_data = reactive(SystemStatus())

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Ensure we have fresh instances for each TuiState
        self.market_data = MarketState()
        self.position_data = PortfolioSummary()
        self.order_data = ActiveOrders()
        self.trade_data = TradeHistory()
        self.account_data = AccountSummary()
        self.strategy_data = StrategyState()
        self.risk_data = RiskState()
        self.system_data = SystemStatus()

    def update_from_bot_status(
        self, status: dict[str, Any], runtime_state: Any | None = None
    ) -> None:
        """
        Update state from the bot's status dictionary.

        This method orchestrates updates for all data components.
        Failures in one component update do not block others.
        """
        self._update_market_data(status.get("market", {}))
        self._update_position_data(status.get("positions", {}))
        self._update_order_data(status.get("orders", []))
        self._update_trade_data(status.get("trades", []))
        self._update_account_data(status.get("account", {}))
        self._update_strategy_data(status.get("strategy", {}))
        self._update_risk_data(status.get("risk", {}))
        self._update_system_data(status.get("system", {}))
        self._update_runtime_stats(runtime_state)

    def _update_market_data(self, market: dict[str, Any]) -> None:
        try:
            self.market_data = MarketState(
                prices=market.get("last_prices", {}),
                last_update=market.get("last_price_update", 0.0),
                price_history=market.get("price_history", {}),
            )
        except Exception:
            # Log error if logging were available, or just suppress to keep UI alive
            pass

    def _update_position_data(self, pos_data: dict[str, Any] | Any) -> None:
        try:
            positions_map = {}
            # Handle different potential structures of 'positions'
            # It might be {symbol: {quantity, ...}} or {symbol: PositionObject}

            # If pos_data is not a dict, it might be an object or list, but we expect dict-like access usually
            # If it's strictly an object without .items(), this loop will fail.
            # Let's assume it's a dict for the iteration.

            if isinstance(pos_data, dict):
                items = pos_data.items()
            else:
                # Fallback or empty if not iterable as dict
                items = []

            for symbol, p_data in items:
                if isinstance(p_data, dict):
                    positions_map[symbol] = Position(
                        symbol=symbol,
                        quantity=str(p_data.get("quantity", "0")),
                        entry_price=str(p_data.get("entry_price", "N/A")),
                        unrealized_pnl=str(p_data.get("unrealized_pnl", "0.00")),
                        mark_price=str(p_data.get("mark_price", "0.00")),
                        side=str(p_data.get("side", "")),
                    )
                elif hasattr(p_data, "quantity"):
                    # Handle object-like position data
                    positions_map[symbol] = Position(
                        symbol=symbol,
                        quantity=str(p_data.quantity),
                        entry_price=str(getattr(p_data, "entry_price", "N/A")),
                        unrealized_pnl=str(getattr(p_data, "unrealized_pnl", "0.00")),
                        mark_price=str(getattr(p_data, "mark_price", "0.00")),
                        side=str(getattr(p_data, "side", "")),
                    )

            total_upnl = "0.00"
            equity = "0.00"

            if isinstance(pos_data, dict):
                total_upnl = str(pos_data.get("total_unrealized_pnl", "0.00"))
                equity = str(pos_data.get("equity", "0.00"))

            self.position_data = PortfolioSummary(
                positions=positions_map,
                total_unrealized_pnl=total_upnl,
                equity=equity,
            )
        except Exception:
            pass

    def _update_order_data(self, raw_orders: list[Any]) -> None:
        try:
            orders_list = []
            for o in raw_orders:
                if isinstance(o, dict):
                    orders_list.append(
                        Order(
                            order_id=str(o.get("order_id", "")),
                            symbol=str(o.get("symbol", "")),
                            side=str(o.get("side", "")),
                            quantity=str(o.get("quantity", "")),
                            price=str(o.get("avg_execution_price") or o.get("price", "")),
                            status=str(o.get("status", "UNKNOWN")),
                            type=str(o.get("order_type", "UNKNOWN")),
                            time_in_force=str(o.get("time_in_force", "UNKNOWN")),
                            creation_time=str(o.get("creation_time", "")),
                        )
                    )
            self.order_data = ActiveOrders(orders=orders_list)
        except Exception:
            pass

    def _update_trade_data(self, raw_trades: list[Any]) -> None:
        try:
            trades_list = []
            for t in raw_trades:
                if isinstance(t, dict):
                    trades_list.append(
                        Trade(
                            trade_id=str(t.get("trade_id", "")),
                            symbol=str(t.get("product_id") or t.get("symbol", "")),
                            side=str(t.get("side", "")),
                            quantity=str(t.get("quantity", "")),
                            price=str(t.get("price", "")),
                            order_id=str(t.get("order_id", "")),
                            time=str(t.get("time", "")),
                            fee=str(t.get("fee", "0.00")),
                        )
                    )
            self.trade_data = TradeHistory(trades=trades_list)
        except Exception:
            pass

    def _update_account_data(self, acc: dict[str, Any]) -> None:
        try:
            raw_balances = acc.get("balances", [])
            balances_list = []
            for b in raw_balances:
                if isinstance(b, dict):
                    balances_list.append(
                        AccountBalance(
                            asset=str(b.get("asset", "")),
                            total=str(b.get("total", "0")),
                            available=str(b.get("available", "0")),
                            hold=str(b.get("hold", "0.00")),
                        )
                    )

            self.account_data = AccountSummary(
                volume_30d=str(acc.get("volume_30d", "0.00")),
                fees_30d=str(acc.get("fees_30d", "0.00")),
                fee_tier=str(acc.get("fee_tier", "")),
                balances=balances_list,
            )
        except Exception:
            pass

    def _update_strategy_data(self, strat: dict[str, Any]) -> None:
        try:
            decisions = {}

            # StatusReporter stores last_decisions as a list of dicts
            raw_decisions = strat.get("last_decisions", [])
            if isinstance(raw_decisions, dict):
                # Handle legacy/alternative format if any
                raw_decisions = list(raw_decisions.values())

            for dec in raw_decisions:
                if not isinstance(dec, dict):
                    continue

                symbol = dec.get("symbol")
                if not symbol:
                    continue

                # Parse fields (they might be strings from StatusReporter)
                try:
                    conf = float(dec.get("confidence", 0.0))
                except (ValueError, TypeError):
                    conf = 0.0

                ts = dec.get("timestamp", 0.0)
                if isinstance(ts, str):
                    try:
                        ts = float(ts)
                    except ValueError:
                        ts = 0.0

                decisions[symbol] = DecisionData(
                    symbol=symbol,
                    action=dec.get("action", "HOLD"),
                    reason=dec.get("reason", ""),
                    confidence=conf,
                    indicators=dec.get("indicators", {}),
                    timestamp=ts,
                )

            self.strategy_data = StrategyState(
                active_strategies=strat.get("active_strategies", []),
                last_decisions=decisions,
            )
        except Exception:
            pass

    def _update_risk_data(self, risk: dict[str, Any]) -> None:
        try:
            self.risk_data = RiskState(
                max_leverage=risk.get("max_leverage", 0.0),
                daily_loss_limit_pct=risk.get("daily_loss_limit_pct", 0.0),
                current_daily_loss_pct=risk.get("current_daily_loss_pct", 0.0),
                reduce_only_mode=risk.get("reduce_only_mode", False),
                reduce_only_reason=risk.get("reduce_only_reason", ""),
                active_guards=risk.get("active_guards", []),
            )
        except Exception:
            pass

    def _update_system_data(self, sys: dict[str, Any]) -> None:
        try:
            self.system_data = SystemStatus(
                api_latency=sys.get("api_latency", 0.0),
                connection_status=sys.get("connection_status", "UNKNOWN"),
                rate_limit_usage=sys.get("rate_limit_usage", "0%"),
                memory_usage=sys.get("memory_usage", "0MB"),
                cpu_usage=sys.get("cpu_usage", "0%"),
            )
        except Exception:
            pass

    def _update_runtime_stats(self, runtime_state: Any | None) -> None:
        try:
            if runtime_state:
                self.uptime = runtime_state.uptime
                # self.cycle_count = runtime_state.cycle_count # If available
        except Exception:
            pass
