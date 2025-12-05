"""
TUI State Management.
"""

from __future__ import annotations

from typing import Any

from textual.reactive import reactive
from textual.widget import Widget

from gpt_trader.monitoring.status_reporter import BotStatus
from gpt_trader.tui.formatting import safe_decimal
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
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class TuiState(Widget):
    """
    Reactive state for the TUI.
    Acts as a ViewModel that the App observes.
    """

    # Reactive properties that widgets can watch
    running = reactive(False)
    uptime = reactive(0.0)
    cycle_count = reactive(0)

    # Mode and connection tracking
    data_source_mode = reactive("demo")  # demo, paper, read_only, live
    last_update_timestamp = reactive(0.0)
    update_interval = reactive(2.0)
    connection_healthy = reactive(True)

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

    def update_from_bot_status(self, status: BotStatus, runtime_state: Any | None = None) -> None:
        """
        Update state from the bot's typed status snapshot.

        This method orchestrates updates for all data components using typed
        dataclasses, eliminating the need for defensive parsing.

        Args:
            status: Typed BotStatus snapshot from StatusReporter
            runtime_state: Optional runtime state (engine state, uptime, etc.)
        """
        self._update_market_data(status.market)
        self._update_position_data(status.positions)
        self._update_order_data(status.orders)
        self._update_trade_data(status.trades)
        self._update_account_data(status.account)
        self._update_strategy_data(status.strategy)
        self._update_risk_data(status.risk)
        self._update_system_data(status.system)
        self._update_runtime_stats(runtime_state)

    def check_connection_health(self) -> bool:
        """
        Check if data connection is healthy based on update interval.

        Returns:
            True if data is fresh, False if stale
        """
        if self.data_source_mode == "demo":
            return True  # Demo always healthy

        import time

        time_since_update = time.time() - self.last_update_timestamp
        staleness_threshold = self.update_interval * 2.5
        is_healthy = time_since_update < staleness_threshold

        # Update connection_healthy if changed
        if is_healthy != self.connection_healthy:
            self.connection_healthy = is_healthy

        return is_healthy

    def _update_market_data(self, market: Any) -> None:  # MarketStatus from status_reporter
        """Update market data from typed MarketStatus."""
        # Track update timestamp for connection health monitoring
        # Use the actual market data timestamp, not receipt time
        if hasattr(market, "last_price_update") and market.last_price_update:
            self.last_update_timestamp = market.last_price_update
        else:
            # Fallback to current time if no timestamp available
            import time

            self.last_update_timestamp = time.time()

        # Convert prices to Decimal (StatusReporter may provide str or already Decimal)
        prices_decimal = {}
        if hasattr(market, "last_prices") and market.last_prices:
            for symbol, price in market.last_prices.items():
                prices_decimal[symbol] = safe_decimal(price)

        # Convert price history to Decimal
        price_history_converted = {}
        if hasattr(market, "price_history") and market.price_history:
            for symbol, history in market.price_history.items():
                if isinstance(history, list):
                    price_history_converted[symbol] = [safe_decimal(p) for p in history]
                else:
                    price_history_converted[symbol] = history

        self.market_data = MarketState(
            prices=prices_decimal,
            last_update=market.last_price_update if hasattr(market, "last_price_update") else 0.0,
            price_history=price_history_converted,
        )

    def _update_position_data(self, pos_data: Any) -> None:  # PositionStatus from status_reporter
        """Update position data from typed PositionStatus."""
        positions_map = {}

        # PositionStatus has positions dict that contains position data
        # Each position is a dict with keys: quantity, entry_price, unrealized_pnl, mark_price, side
        for symbol, p_data in pos_data.positions.items():
            if isinstance(p_data, dict):
                positions_map[symbol] = Position(
                    symbol=symbol,
                    quantity=safe_decimal(p_data.get("quantity", "0")),
                    entry_price=safe_decimal(p_data.get("entry_price", "0")),
                    unrealized_pnl=safe_decimal(p_data.get("unrealized_pnl", "0")),
                    mark_price=safe_decimal(p_data.get("mark_price", "0")),
                    side=str(p_data.get("side", "")),
                )

        self.position_data = PortfolioSummary(
            positions=positions_map,
            total_unrealized_pnl=safe_decimal(pos_data.total_unrealized_pnl),
            equity=safe_decimal(pos_data.equity),
        )

    def _update_order_data(self, raw_orders: list[Any]) -> None:  # list[OrderStatus]
        """Update order data from typed list of OrderStatus."""
        orders_list = []
        for o in raw_orders:
            # OrderStatus from status_reporter has order_type field (not type)
            orders_list.append(
                Order(
                    order_id=o.order_id,
                    symbol=o.symbol,
                    side=o.side,
                    quantity=safe_decimal(o.quantity),
                    price=safe_decimal(o.price) if o.price else safe_decimal("0"),
                    status=o.status,
                    type=o.order_type,  # OrderStatus uses order_type field
                    time_in_force=o.time_in_force,
                    creation_time=str(o.creation_time) if o.creation_time else "",
                )
            )
        self.order_data = ActiveOrders(orders=orders_list)

    def _update_trade_data(self, raw_trades: list[Any]) -> None:  # list[TradeStatus]
        """Update trade data from typed list of TradeStatus."""
        trades_list = []
        for t in raw_trades:
            # TradeStatus from status_reporter has all fields typed
            trades_list.append(
                Trade(
                    trade_id=t.trade_id,
                    symbol=t.symbol,
                    side=t.side,
                    quantity=safe_decimal(t.quantity),
                    price=safe_decimal(t.price),
                    order_id=t.order_id,
                    time=t.time,
                    fee=safe_decimal(t.fee),
                )
            )
        self.trade_data = TradeHistory(trades=trades_list)

    def _update_account_data(self, acc: Any) -> None:  # AccountStatus from status_reporter
        """Update account data from typed AccountStatus."""
        balances_list = []
        for b in acc.balances:
            # BalanceEntry from StatusReporter → AccountBalance for TUI
            # Both have identical fields (asset, total, available, hold as Decimal)
            if hasattr(b, "asset"):
                balances_list.append(
                    AccountBalance(
                        asset=b.asset,
                        total=b.total,  # Already Decimal from BalanceEntry
                        available=b.available,
                        hold=b.hold,
                    )
                )

        self.account_data = AccountSummary(
            volume_30d=acc.volume_30d,  # Already Decimal from AccountStatus
            fees_30d=acc.fees_30d,
            fee_tier=acc.fee_tier,
            balances=balances_list,
        )

    def _update_strategy_data(self, strat: Any) -> None:  # StrategyStatus from status_reporter
        """Update strategy data from typed StrategyStatus with DecisionEntry objects."""
        decisions = {}

        # StrategyStatus stores last_decisions as list[DecisionEntry] (already typed)
        for dec in strat.last_decisions:
            # DecisionEntry from StatusReporter has all fields typed
            if not hasattr(dec, "symbol") or not dec.symbol:
                continue

            decisions[dec.symbol] = DecisionData(
                symbol=dec.symbol,
                action=dec.action,
                reason=dec.reason,
                confidence=dec.confidence,  # Already float from DecisionEntry
                indicators=dec.indicators,
                timestamp=dec.timestamp,  # Already float from DecisionEntry
            )

        self.strategy_data = StrategyState(
            active_strategies=strat.active_strategies,
            last_decisions=decisions,
        )

    def _update_risk_data(self, risk: Any) -> None:  # RiskStatus from status_reporter
        """Update risk data from typed RiskStatus."""
        # Note: RiskStatus doesn't have position_leverage field, but TUI RiskState does
        # We need to add it to RiskStatus or handle it separately
        self.risk_data = RiskState(
            max_leverage=risk.max_leverage,
            daily_loss_limit_pct=risk.daily_loss_limit_pct,
            current_daily_loss_pct=risk.current_daily_loss_pct,
            reduce_only_mode=risk.reduce_only_mode,
            reduce_only_reason=risk.reduce_only_reason,
            active_guards=risk.active_guards,
            position_leverage={},  # Will be populated separately if needed
        )

    def _update_system_data(self, sys: Any) -> None:  # SystemStatus from status_reporter
        """Update system data from typed SystemStatus."""
        self.system_data = SystemStatus(
            api_latency=sys.api_latency,
            connection_status=sys.connection_status,
            rate_limit_usage=sys.rate_limit_usage,
            memory_usage=sys.memory_usage,
            cpu_usage=sys.cpu_usage,
        )

    def _update_runtime_stats(self, runtime_state: Any | None) -> None:
        """Update runtime statistics (uptime, cycle count, etc.)."""
        if runtime_state and hasattr(runtime_state, "uptime"):
            self.uptime = runtime_state.uptime
        if runtime_state and hasattr(runtime_state, "cycle_count"):
            self.cycle_count = runtime_state.cycle_count
