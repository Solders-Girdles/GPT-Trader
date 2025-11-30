"""
TUI State Management.
"""

from __future__ import annotations

from typing import Any

from textual.reactive import reactive
from textual.widget import Widget

from gpt_trader.tui.models import (
    AccountData,
    DecisionData,
    MarketData,
    OrderData,
    PositionData,
    RiskData,
    StrategyData,
    SystemData,
    TradeData,
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
    market_data = reactive(MarketData())
    position_data = reactive(PositionData())
    order_data = reactive(OrderData())
    trade_data = reactive(TradeData())
    account_data = reactive(AccountData())
    strategy_data = reactive(StrategyData())
    risk_data = reactive(RiskData())
    system_data = reactive(SystemData())

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Ensure we have fresh instances for each TuiState
        self.market_data = MarketData()
        self.position_data = PositionData()
        self.order_data = OrderData()
        self.trade_data = TradeData()
        self.account_data = AccountData()
        self.strategy_data = StrategyData()
        self.risk_data = RiskData()
        self.system_data = SystemData()

    def update_from_bot_status(
        self, status: dict[str, Any], runtime_state: Any | None = None
    ) -> None:
        """Update state from the bot's status dictionary."""

        # Update Market Data
        market = status.get("market", {})
        self.market_data = MarketData(
            prices=market.get("last_prices", {}),
            last_update=market.get("last_price_update", 0.0),
            price_history=market.get("price_history", {}),
        )

        # Update Position Data
        pos = status.get("positions", {})
        self.position_data = PositionData(
            positions=pos,  # This might need processing depending on structure
            total_unrealized_pnl=pos.get("total_unrealized_pnl", "0.00"),
            equity=pos.get("equity", "0.00"),
        )

        # Update Order Data
        self.order_data = OrderData(orders=status.get("orders", []))

        # Update Trade Data
        self.trade_data = TradeData(trades=status.get("trades", []))

        # Update Account Data
        acc = status.get("account", {})
        self.account_data = AccountData(
            volume_30d=acc.get("volume_30d", "0.00"),
            fees_30d=acc.get("fees_30d", "0.00"),
            fee_tier=acc.get("fee_tier", ""),
            balances=acc.get("balances", []),
        )

        # Update Strategy Data
        strat = status.get("strategy", {})
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

        self.strategy_data = StrategyData(
            active_strategies=strat.get("active_strategies", []),
            last_decisions=decisions,
        )

        # Update Risk Data
        risk = status.get("risk", {})
        self.risk_data = RiskData(
            max_leverage=risk.get("max_leverage", 0.0),
            daily_loss_limit_pct=risk.get("daily_loss_limit_pct", 0.0),
            current_daily_loss_pct=risk.get("current_daily_loss_pct", 0.0),
            reduce_only_mode=risk.get("reduce_only_mode", False),
            reduce_only_reason=risk.get("reduce_only_reason", ""),
            active_guards=risk.get("active_guards", []),
        )

        # Update System Data
        sys = status.get("system", {})
        self.system_data = SystemData(
            api_latency=sys.get("api_latency", 0.0),
            connection_status=sys.get("connection_status", "UNKNOWN"),
            rate_limit_usage=sys.get("rate_limit_usage", "0%"),
            memory_usage=sys.get("memory_usage", "0MB"),
            cpu_usage=sys.get("cpu_usage", "0%"),
        )

        # Update Runtime Stats
        if runtime_state:
            self.uptime = runtime_state.uptime
            # self.cycle_count = runtime_state.cycle_count # If available
