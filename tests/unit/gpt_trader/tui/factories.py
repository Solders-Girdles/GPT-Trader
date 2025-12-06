"""
TUI Test Factories.

Provides factory classes for creating test data objects with sensible defaults.
Enables quick creation of valid test fixtures for TUI testing.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from gpt_trader.monitoring.status_reporter import (
    AccountStatus,
    BalanceEntry,
    BotStatus,
    DecisionEntry,
    EngineStatus,
    HeartbeatStatus,
    MarketStatus,
    OrderStatus,
    PositionStatus,
    RiskStatus,
    StrategyStatus,
    TradeStatus,
)
from gpt_trader.monitoring.status_reporter import (
    SystemStatus as ReporterSystemStatus,
)
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import (
    ActiveOrders,
    MarketState,
    Order,
    PortfolioSummary,
    Position,
    Trade,
    TradeHistory,
)


class BotStatusFactory:
    """Factory for creating BotStatus test fixtures."""

    @staticmethod
    def create_minimal() -> BotStatus:
        """Create a minimal valid BotStatus for quick tests."""
        return BotStatus(
            bot_id="test-bot",
            timestamp=time.time(),
            version="1.0.0",
            healthy=True,
        )

    @staticmethod
    def create_running(
        uptime: float = 120.0,
        cycle_count: int = 50,
    ) -> BotStatus:
        """Create a BotStatus representing a running bot."""
        return BotStatus(
            bot_id="test-bot",
            timestamp=time.time(),
            engine=EngineStatus(
                running=True,
                uptime_seconds=uptime,
                cycle_count=cycle_count,
            ),
            healthy=True,
        )

    @staticmethod
    def create_stopped() -> BotStatus:
        """Create a BotStatus representing a stopped bot."""
        return BotStatus(
            bot_id="test-bot",
            timestamp=time.time(),
            engine=EngineStatus(running=False),
            healthy=True,
        )

    @staticmethod
    def create_with_positions(
        positions: dict[str, dict[str, Any]] | None = None,
        equity: Decimal = Decimal("10000.00"),
    ) -> BotStatus:
        """Create a BotStatus with specific positions for P&L testing.

        Args:
            positions: Position data as dict[symbol, {quantity, entry_price, ...}]
            equity: Total equity amount
        """
        if positions is None:
            positions = {
                "BTC-USD": {
                    "quantity": Decimal("0.5"),
                    "entry_price": Decimal("42000.00"),
                    "unrealized_pnl": Decimal("500.00"),
                    "mark_price": Decimal("43000.00"),
                    "side": "LONG",
                },
                "ETH-USD": {
                    "quantity": Decimal("2.0"),
                    "entry_price": Decimal("2800.00"),
                    "unrealized_pnl": Decimal("-50.00"),
                    "mark_price": Decimal("2775.00"),
                    "side": "LONG",
                },
            }

        total_pnl = sum(p.get("unrealized_pnl", Decimal("0")) for p in positions.values())

        return BotStatus(
            bot_id="test-bot",
            timestamp=time.time(),
            engine=EngineStatus(running=True),
            positions=PositionStatus(
                count=len(positions),
                symbols=list(positions.keys()),
                total_unrealized_pnl=total_pnl,
                equity=equity,
                positions=positions,
            ),
            market=MarketStatus(
                symbols=list(positions.keys()),
                last_prices={
                    sym: pos.get("mark_price", Decimal("0")) for sym, pos in positions.items()
                },
            ),
            healthy=True,
        )

    @staticmethod
    def create_with_orders(
        pending_orders: list[OrderStatus] | None = None,
    ) -> BotStatus:
        """Create a BotStatus with pending orders."""
        if pending_orders is None:
            pending_orders = [
                OrderStatus(
                    order_id="order-001",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("0.1"),
                    price=Decimal("40000.00"),
                    status="PENDING",
                    order_type="LIMIT",
                ),
                OrderStatus(
                    order_id="order-002",
                    symbol="ETH-USD",
                    side="SELL",
                    quantity=Decimal("1.0"),
                    price=Decimal("3000.00"),
                    status="PENDING",
                    order_type="LIMIT",
                ),
            ]

        return BotStatus(
            bot_id="test-bot",
            timestamp=time.time(),
            engine=EngineStatus(running=True),
            orders=pending_orders,
            healthy=True,
        )

    @staticmethod
    def create_with_trades(
        trades: list[TradeStatus] | None = None,
    ) -> BotStatus:
        """Create a BotStatus with recent trades."""
        if trades is None:
            trades = [
                TradeStatus(
                    trade_id="trade-001",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("0.5"),
                    price=Decimal("42000.00"),
                    time="2024-01-15T10:30:00Z",
                    order_id="order-001",
                    fee=Decimal("10.50"),
                ),
                TradeStatus(
                    trade_id="trade-002",
                    symbol="BTC-USD",
                    side="SELL",
                    quantity=Decimal("0.5"),
                    price=Decimal("43000.00"),
                    time="2024-01-15T11:00:00Z",
                    order_id="order-002",
                    fee=Decimal("10.75"),
                ),
            ]

        return BotStatus(
            bot_id="test-bot",
            timestamp=time.time(),
            engine=EngineStatus(running=True),
            trades=trades,
            healthy=True,
        )

    @staticmethod
    def create_live_scenario(
        equity: Decimal = Decimal("50000.00"),
        daily_pnl_pct: float = 0.02,
    ) -> BotStatus:
        """Create a realistic live trading scenario for integration tests.

        Args:
            equity: Account equity
            daily_pnl_pct: Current daily P&L as decimal (0.02 = 2%)
        """
        positions = {
            "BTC-USD": {
                "quantity": Decimal("1.0"),
                "entry_price": Decimal("42000.00"),
                "unrealized_pnl": Decimal("1200.00"),
                "mark_price": Decimal("43200.00"),
                "side": "LONG",
            },
        }

        return BotStatus(
            bot_id="live-trader-001",
            timestamp=time.time(),
            engine=EngineStatus(
                running=True,
                uptime_seconds=3600.0,
                cycle_count=1800,
            ),
            market=MarketStatus(
                symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
                last_prices={
                    "BTC-USD": Decimal("43200.00"),
                    "ETH-USD": Decimal("2850.00"),
                    "SOL-USD": Decimal("95.50"),
                },
                price_history={
                    "BTC-USD": [
                        Decimal("42800"),
                        Decimal("42900"),
                        Decimal("43000"),
                        Decimal("43100"),
                        Decimal("43200"),
                    ],
                },
            ),
            positions=PositionStatus(
                count=1,
                symbols=["BTC-USD"],
                total_unrealized_pnl=Decimal("1200.00"),
                equity=equity,
                positions=positions,
            ),
            account=AccountStatus(
                volume_30d=Decimal("250000.00"),
                fees_30d=Decimal("125.00"),
                fee_tier="Maker: 0.04%, Taker: 0.06%",
                balances=[
                    BalanceEntry(
                        asset="USD",
                        total=Decimal("48800.00"),
                        available=Decimal("48800.00"),
                    ),
                    BalanceEntry(
                        asset="BTC",
                        total=Decimal("1.0"),
                        available=Decimal("1.0"),
                    ),
                ],
            ),
            strategy=StrategyStatus(
                active_strategies=["momentum", "mean_reversion"],
                last_decisions=[
                    DecisionEntry(
                        symbol="BTC-USD",
                        action="HOLD",
                        reason="Strong momentum, maintaining position",
                        confidence=0.75,
                        timestamp=time.time(),
                    ),
                ],
            ),
            risk=RiskStatus(
                max_leverage=3.0,
                daily_loss_limit_pct=0.05,
                current_daily_loss_pct=daily_pnl_pct,
                reduce_only_mode=False,
            ),
            system=ReporterSystemStatus(
                api_latency=0.045,
                connection_status="CONNECTED",
                rate_limit_usage="12%",
                memory_usage="256MB",
                cpu_usage="8%",
            ),
            heartbeat=HeartbeatStatus(
                enabled=True,
                running=True,
                is_healthy=True,
                heartbeat_count=120,
            ),
            healthy=True,
        )

    @staticmethod
    def create_unhealthy(
        issues: list[str] | None = None,
    ) -> BotStatus:
        """Create a BotStatus with health issues."""
        if issues is None:
            issues = ["High API latency", "Rate limit approaching"]

        return BotStatus(
            bot_id="test-bot",
            timestamp=time.time(),
            engine=EngineStatus(running=True),
            system=ReporterSystemStatus(
                api_latency=2.5,
                connection_status="DEGRADED",
                rate_limit_usage="85%",
            ),
            healthy=False,
            health_issues=issues,
        )


class TuiStateFactory:
    """Factory for creating TuiState test fixtures."""

    @staticmethod
    def create_default() -> TuiState:
        """Create a default TuiState with no data."""
        return TuiState(validation_enabled=False, delta_updates_enabled=False)

    @staticmethod
    def create_running(
        uptime: float = 300.0,
        cycle_count: int = 150,
        mode: str = "paper",
    ) -> TuiState:
        """Create a TuiState representing a running bot."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.running = True
        state.uptime = uptime
        state.cycle_count = cycle_count
        state.data_source_mode = mode
        state.connection_healthy = True
        return state

    @staticmethod
    def create_stopped(mode: str = "demo") -> TuiState:
        """Create a TuiState representing a stopped bot."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.running = False
        state.uptime = 0.0
        state.cycle_count = 0
        state.data_source_mode = mode
        return state

    @staticmethod
    def create_with_positions(
        positions: dict[str, Position] | None = None,
        total_pnl: Decimal = Decimal("450.00"),
        equity: Decimal = Decimal("10450.00"),
    ) -> TuiState:
        """Create a TuiState with positions loaded."""
        if positions is None:
            positions = {
                "BTC-USD": Position(
                    symbol="BTC-USD",
                    quantity=Decimal("0.5"),
                    entry_price=Decimal("42000.00"),
                    unrealized_pnl=Decimal("500.00"),
                    mark_price=Decimal("43000.00"),
                    side="LONG",
                ),
                "ETH-USD": Position(
                    symbol="ETH-USD",
                    quantity=Decimal("2.0"),
                    entry_price=Decimal("2800.00"),
                    unrealized_pnl=Decimal("-50.00"),
                    mark_price=Decimal("2775.00"),
                    side="LONG",
                ),
            }

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.running = True
        state.position_data = PortfolioSummary(
            positions=positions,
            total_unrealized_pnl=total_pnl,
            equity=equity,
        )
        return state

    @staticmethod
    def create_with_market_data(
        prices: dict[str, Decimal] | None = None,
        price_history: dict[str, list[Decimal]] | None = None,
    ) -> TuiState:
        """Create a TuiState with market data loaded."""
        if prices is None:
            prices = {
                "BTC-USD": Decimal("43000.00"),
                "ETH-USD": Decimal("2850.00"),
                "SOL-USD": Decimal("95.00"),
            }

        if price_history is None:
            price_history = {
                "BTC-USD": [
                    Decimal("42500"),
                    Decimal("42700"),
                    Decimal("42900"),
                    Decimal("43000"),
                    Decimal("43000"),
                ],
            }

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.running = True
        state.market_data = MarketState(
            prices=prices,
            last_update=time.time(),
            price_history=price_history,
        )
        return state

    @staticmethod
    def create_with_orders(
        orders: list[Order] | None = None,
    ) -> TuiState:
        """Create a TuiState with pending orders."""
        if orders is None:
            orders = [
                Order(
                    order_id="order-001",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("0.1"),
                    price=Decimal("40000.00"),
                    status="PENDING",
                    type="LIMIT",
                    time_in_force="GTC",
                ),
            ]

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.running = True
        state.order_data = ActiveOrders(orders=orders)
        return state

    @staticmethod
    def create_with_trades(
        trades: list[Trade] | None = None,
    ) -> TuiState:
        """Create a TuiState with trade history."""
        if trades is None:
            trades = [
                Trade(
                    trade_id="trade-001",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("0.5"),
                    price=Decimal("42000.00"),
                    order_id="order-001",
                    time="2024-01-15T10:30:00Z",
                    fee=Decimal("10.50"),
                ),
            ]

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.running = True
        state.trade_data = TradeHistory(trades=trades)
        return state

    @staticmethod
    def create_with_validation_errors(
        error_count: int = 2,
        warning_count: int = 1,
    ) -> TuiState:
        """Create a TuiState with validation error counts."""
        state = TuiState(validation_enabled=True, delta_updates_enabled=False)
        state.validation_error_count = error_count
        state.validation_warning_count = warning_count
        return state

    @staticmethod
    def create_disconnected() -> TuiState:
        """Create a TuiState with connection issues."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.running = True
        state.connection_healthy = False
        state.last_update_timestamp = time.time() - 30.0  # 30 seconds ago
        state.update_interval = 2.0
        return state


class MarketDataFactory:
    """Factory for creating market data test fixtures."""

    @staticmethod
    def create_minimal() -> MarketState:
        """Create minimal market data."""
        return MarketState(
            prices={"BTC-USD": Decimal("42000.00")},
            last_update=time.time(),
        )

    @staticmethod
    def create_multi_symbol(
        symbols: list[str] | None = None,
    ) -> MarketState:
        """Create market data for multiple symbols."""
        if symbols is None:
            symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]

        prices = {
            "BTC-USD": Decimal("43000.00"),
            "ETH-USD": Decimal("2850.00"),
            "SOL-USD": Decimal("95.00"),
            "DOGE-USD": Decimal("0.12"),
        }

        return MarketState(
            prices={sym: prices.get(sym, Decimal("100.00")) for sym in symbols},
            last_update=time.time(),
            price_history={sym: [prices.get(sym, Decimal("100.00"))] * 5 for sym in symbols},
        )

    @staticmethod
    def create_with_trend(
        symbol: str = "BTC-USD",
        start_price: Decimal = Decimal("40000.00"),
        trend: str = "up",  # "up", "down", "flat"
        periods: int = 10,
    ) -> MarketState:
        """Create market data with a price trend for sparkline testing.

        Args:
            symbol: Trading symbol
            start_price: Starting price for trend
            trend: Direction of trend ("up", "down", "flat")
            periods: Number of price points in history
        """
        if trend == "up":
            increment = Decimal("100.00")
        elif trend == "down":
            increment = Decimal("-100.00")
        else:
            increment = Decimal("0.00")

        history = [start_price + (increment * i) for i in range(periods)]
        current_price = history[-1]

        return MarketState(
            prices={symbol: current_price},
            last_update=time.time(),
            price_history={symbol: history},
        )


class PositionFactory:
    """Factory for creating position test fixtures."""

    @staticmethod
    def create_long(
        symbol: str = "BTC-USD",
        quantity: Decimal = Decimal("1.0"),
        entry_price: Decimal = Decimal("42000.00"),
        current_price: Decimal = Decimal("43000.00"),
    ) -> Position:
        """Create a long position."""
        pnl = (current_price - entry_price) * quantity
        return Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            unrealized_pnl=pnl,
            mark_price=current_price,
            side="LONG",
        )

    @staticmethod
    def create_short(
        symbol: str = "BTC-USD",
        quantity: Decimal = Decimal("1.0"),
        entry_price: Decimal = Decimal("43000.00"),
        current_price: Decimal = Decimal("42000.00"),
    ) -> Position:
        """Create a short position."""
        pnl = (entry_price - current_price) * quantity
        return Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            unrealized_pnl=pnl,
            mark_price=current_price,
            side="SHORT",
        )

    @staticmethod
    def create_profitable(
        symbol: str = "BTC-USD",
        profit_pct: float = 0.05,
    ) -> Position:
        """Create a position with specified profit percentage."""
        entry_price = Decimal("42000.00")
        current_price = entry_price * (1 + Decimal(str(profit_pct)))
        quantity = Decimal("1.0")
        pnl = (current_price - entry_price) * quantity

        return Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            unrealized_pnl=pnl,
            mark_price=current_price,
            side="LONG",
        )

    @staticmethod
    def create_losing(
        symbol: str = "BTC-USD",
        loss_pct: float = 0.03,
    ) -> Position:
        """Create a position with specified loss percentage."""
        entry_price = Decimal("42000.00")
        current_price = entry_price * (1 - Decimal(str(loss_pct)))
        quantity = Decimal("1.0")
        pnl = (current_price - entry_price) * quantity

        return Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            unrealized_pnl=pnl,
            mark_price=current_price,
            side="LONG",
        )


class OrderFactory:
    """Factory for creating order test fixtures."""

    @staticmethod
    def create_buy_limit(
        symbol: str = "BTC-USD",
        quantity: Decimal = Decimal("0.1"),
        price: Decimal = Decimal("40000.00"),
        order_id: str = "order-001",
    ) -> Order:
        """Create a buy limit order."""
        return Order(
            order_id=order_id,
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            price=price,
            status="PENDING",
            type="LIMIT",
            time_in_force="GTC",
        )

    @staticmethod
    def create_sell_limit(
        symbol: str = "BTC-USD",
        quantity: Decimal = Decimal("0.1"),
        price: Decimal = Decimal("45000.00"),
        order_id: str = "order-002",
    ) -> Order:
        """Create a sell limit order."""
        return Order(
            order_id=order_id,
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            price=price,
            status="PENDING",
            type="LIMIT",
            time_in_force="GTC",
        )

    @staticmethod
    def create_market_order(
        symbol: str = "BTC-USD",
        side: str = "BUY",
        quantity: Decimal = Decimal("0.1"),
        order_id: str = "order-003",
    ) -> Order:
        """Create a market order."""
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=Decimal("0"),  # Market orders have no price
            status="PENDING",
            type="MARKET",
            time_in_force="IOC",
        )


class TradeFactory:
    """Factory for creating trade test fixtures."""

    @staticmethod
    def create_buy(
        symbol: str = "BTC-USD",
        quantity: Decimal = Decimal("0.5"),
        price: Decimal = Decimal("42000.00"),
        trade_id: str = "trade-001",
        fee: Decimal = Decimal("10.50"),
    ) -> Trade:
        """Create a buy trade."""
        return Trade(
            trade_id=trade_id,
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            price=price,
            order_id=f"order-{trade_id}",
            time="2024-01-15T10:30:00Z",
            fee=fee,
        )

    @staticmethod
    def create_sell(
        symbol: str = "BTC-USD",
        quantity: Decimal = Decimal("0.5"),
        price: Decimal = Decimal("43000.00"),
        trade_id: str = "trade-002",
        fee: Decimal = Decimal("10.75"),
    ) -> Trade:
        """Create a sell trade."""
        return Trade(
            trade_id=trade_id,
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            price=price,
            order_id=f"order-{trade_id}",
            time="2024-01-15T11:00:00Z",
            fee=fee,
        )

    @staticmethod
    def create_matched_pair(
        symbol: str = "BTC-USD",
        buy_price: Decimal = Decimal("42000.00"),
        sell_price: Decimal = Decimal("43000.00"),
        quantity: Decimal = Decimal("0.5"),
    ) -> tuple[Trade, Trade]:
        """Create a matched buy/sell trade pair for P&L testing."""
        buy_trade = Trade(
            trade_id="trade-buy-001",
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            price=buy_price,
            order_id="order-buy-001",
            time="2024-01-15T10:00:00Z",
            fee=Decimal("10.50"),
        )
        sell_trade = Trade(
            trade_id="trade-sell-001",
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            price=sell_price,
            order_id="order-sell-001",
            time="2024-01-15T11:00:00Z",
            fee=Decimal("10.75"),
        )
        return buy_trade, sell_trade
