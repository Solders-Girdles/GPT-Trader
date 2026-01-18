"""Tests for StateDeltaUpdater."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.state_management.delta_updater import DeltaResult, StateDeltaUpdater
from gpt_trader.tui.types import (
    AccountBalance,
    AccountSummary,
    ActiveOrders,
    MarketState,
    Order,
    PortfolioSummary,
    Position,
    RiskGuard,
    RiskState,
    SystemStatus,
    Trade,
    TradeHistory,
)


class TestDeltaResult:
    """Test DeltaResult dataclass."""

    def test_initial_state_no_changes(self):
        """Test new DeltaResult has no changes."""
        result = DeltaResult()

        assert result.has_changes is False
        assert result.changed_fields == []
        assert result.details == {}

    def test_add_change_sets_has_changes(self):
        """Test adding a change sets has_changes flag."""
        result = DeltaResult()
        result.add_change("field", "old", "new")

        assert result.has_changes is True
        assert "field" in result.changed_fields
        assert result.details["field"] == ("old", "new")

    def test_add_multiple_changes(self):
        """Test adding multiple changes."""
        result = DeltaResult()
        result.add_change("field1", "old1", "new1")
        result.add_change("field2", "old2", "new2")

        assert len(result.changed_fields) == 2
        assert len(result.details) == 2


class TestStateDeltaUpdater:
    """Test StateDeltaUpdater functionality."""

    def test_compare_market_no_changes(self):
        """Test market comparison with identical data."""
        updater = StateDeltaUpdater()
        market = MarketState(
            prices={"BTC-USD": Decimal("50000")},
            last_update=1234567890.0,
        )

        result = updater.compare_market(market, market)

        assert not result.has_changes

    def test_compare_market_price_change(self):
        """Test market comparison detects price change."""
        updater = StateDeltaUpdater()
        old_market = MarketState(
            prices={"BTC-USD": Decimal("50000")},
            last_update=1234567890.0,
        )
        new_market = MarketState(
            prices={"BTC-USD": Decimal("51000")},
            last_update=1234567891.0,
        )

        result = updater.compare_market(old_market, new_market)

        assert result.has_changes
        assert "prices.BTC-USD" in result.changed_fields

    def test_compare_market_new_symbol(self):
        """Test market comparison detects new symbol."""
        updater = StateDeltaUpdater()
        old_market = MarketState(
            prices={"BTC-USD": Decimal("50000")},
            last_update=1234567890.0,
        )
        new_market = MarketState(
            prices={"BTC-USD": Decimal("50000"), "ETH-USD": Decimal("3000")},
            last_update=1234567890.0,
        )

        result = updater.compare_market(old_market, new_market)

        assert result.has_changes
        assert "prices.ETH-USD" in result.changed_fields

    def test_compare_market_removed_symbol(self):
        """Test market comparison detects removed symbol."""
        updater = StateDeltaUpdater()
        old_market = MarketState(
            prices={"BTC-USD": Decimal("50000"), "ETH-USD": Decimal("3000")},
            last_update=1234567890.0,
        )
        new_market = MarketState(
            prices={"BTC-USD": Decimal("50000")},
            last_update=1234567890.0,
        )

        result = updater.compare_market(old_market, new_market)

        assert result.has_changes
        assert "prices.ETH-USD" in result.changed_fields

    def test_compare_positions_no_changes(self):
        """Test position comparison with identical data."""
        updater = StateDeltaUpdater()
        positions = PortfolioSummary(
            positions={
                "BTC-USD": Position(
                    symbol="BTC-USD",
                    quantity=Decimal("1.0"),
                    entry_price=Decimal("50000"),
                    unrealized_pnl=Decimal("1000"),
                    mark_price=Decimal("51000"),
                )
            },
            total_unrealized_pnl=Decimal("1000"),
            equity=Decimal("51000"),
        )

        result = updater.compare_positions(positions, positions)

        assert not result.has_changes

    def test_compare_positions_pnl_change(self):
        """Test position comparison detects P&L change."""
        updater = StateDeltaUpdater()
        old_positions = PortfolioSummary(
            positions={
                "BTC-USD": Position(
                    symbol="BTC-USD",
                    quantity=Decimal("1.0"),
                    entry_price=Decimal("50000"),
                    unrealized_pnl=Decimal("1000"),
                    mark_price=Decimal("51000"),
                )
            },
            total_unrealized_pnl=Decimal("1000"),
            equity=Decimal("51000"),
        )
        new_positions = PortfolioSummary(
            positions={
                "BTC-USD": Position(
                    symbol="BTC-USD",
                    quantity=Decimal("1.0"),
                    entry_price=Decimal("50000"),
                    unrealized_pnl=Decimal("2000"),
                    mark_price=Decimal("52000"),
                )
            },
            total_unrealized_pnl=Decimal("2000"),
            equity=Decimal("52000"),
        )

        result = updater.compare_positions(old_positions, new_positions)

        assert result.has_changes
        assert "positions.BTC-USD.unrealized_pnl" in result.changed_fields

    def test_compare_orders_no_changes(self):
        """Test order comparison with identical data."""
        updater = StateDeltaUpdater()
        orders = ActiveOrders(
            orders=[
                Order(
                    order_id="order-1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000"),
                    status="OPEN",
                )
            ]
        )

        result = updater.compare_orders(orders, orders)

        assert not result.has_changes

    def test_compare_orders_status_change(self):
        """Test order comparison detects status change."""
        updater = StateDeltaUpdater()
        old_orders = ActiveOrders(
            orders=[
                Order(
                    order_id="order-1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000"),
                    status="OPEN",
                )
            ]
        )
        new_orders = ActiveOrders(
            orders=[
                Order(
                    order_id="order-1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000"),
                    status="FILLED",
                )
            ]
        )

        result = updater.compare_orders(old_orders, new_orders)

        assert result.has_changes
        assert "orders.order-1.status" in result.changed_fields

    def test_compare_orders_new_order(self):
        """Test order comparison detects new order."""
        updater = StateDeltaUpdater()
        old_orders = ActiveOrders(orders=[])
        new_orders = ActiveOrders(
            orders=[
                Order(
                    order_id="order-1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000"),
                    status="OPEN",
                )
            ]
        )

        result = updater.compare_orders(old_orders, new_orders)

        assert result.has_changes
        assert "orders.order-1" in result.changed_fields

    def test_compare_trades_no_changes(self):
        """Test trade comparison with identical data."""
        updater = StateDeltaUpdater()
        trades = TradeHistory(
            trades=[
                Trade(
                    trade_id="trade-1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000"),
                    order_id="order-1",
                    time="2024-01-01T00:00:00Z",
                )
            ]
        )

        result = updater.compare_trades(trades, trades)

        assert not result.has_changes

    def test_compare_trades_new_trade(self):
        """Test trade comparison detects new trade."""
        updater = StateDeltaUpdater()
        old_trades = TradeHistory(trades=[])
        new_trades = TradeHistory(
            trades=[
                Trade(
                    trade_id="trade-1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("1.0"),
                    price=Decimal("50000"),
                    order_id="order-1",
                    time="2024-01-01T00:00:00Z",
                )
            ]
        )

        result = updater.compare_trades(old_trades, new_trades)

        assert result.has_changes
        assert "trades.trade-1" in result.changed_fields

    def test_compare_account_no_changes(self):
        """Test account comparison with identical data."""
        updater = StateDeltaUpdater()
        account = AccountSummary(
            volume_30d=Decimal("100000"),
            fees_30d=Decimal("500"),
            fee_tier="Advanced",
            balances=[
                AccountBalance(
                    asset="USD",
                    total=Decimal("10000"),
                    available=Decimal("9000"),
                )
            ],
        )

        result = updater.compare_account(account, account)

        assert not result.has_changes

    def test_compare_account_balance_change(self):
        """Test account comparison detects balance change."""
        updater = StateDeltaUpdater()
        old_account = AccountSummary(
            volume_30d=Decimal("100000"),
            fees_30d=Decimal("500"),
            fee_tier="Advanced",
            balances=[
                AccountBalance(
                    asset="USD",
                    total=Decimal("10000"),
                    available=Decimal("9000"),
                )
            ],
        )
        new_account = AccountSummary(
            volume_30d=Decimal("100000"),
            fees_30d=Decimal("500"),
            fee_tier="Advanced",
            balances=[
                AccountBalance(
                    asset="USD",
                    total=Decimal("11000"),
                    available=Decimal("10000"),
                )
            ],
        )

        result = updater.compare_account(old_account, new_account)

        assert result.has_changes
        assert "balances.USD.total" in result.changed_fields

    def test_compare_risk_no_changes(self):
        """Test risk comparison with identical data."""
        updater = StateDeltaUpdater()
        risk = RiskState(
            max_leverage=10.0,
            daily_loss_limit_pct=5.0,
            current_daily_loss_pct=1.0,
            reduce_only_mode=False,
            reduce_only_reason="",
            guards=[RiskGuard(name="guard1")],
        )

        result = updater.compare_risk(risk, risk)

        assert not result.has_changes

    def test_compare_risk_reduce_only_change(self):
        """Test risk comparison detects reduce_only mode change."""
        updater = StateDeltaUpdater()
        old_risk = RiskState(
            max_leverage=10.0,
            daily_loss_limit_pct=5.0,
            current_daily_loss_pct=1.0,
            reduce_only_mode=False,
            reduce_only_reason="",
        )
        new_risk = RiskState(
            max_leverage=10.0,
            daily_loss_limit_pct=5.0,
            current_daily_loss_pct=6.0,  # Exceeded limit
            reduce_only_mode=True,
            reduce_only_reason="Daily loss limit exceeded",
        )

        result = updater.compare_risk(old_risk, new_risk)

        assert result.has_changes
        assert "reduce_only_mode" in result.changed_fields

    def test_compare_system_no_changes(self):
        """Test system comparison with identical data."""
        updater = StateDeltaUpdater()
        system = SystemStatus(
            api_latency=0.05,
            connection_status="CONNECTED",
            rate_limit_usage="50%",
            memory_usage="100MB",
            cpu_usage="10%",
        )

        result = updater.compare_system(system, system)

        assert not result.has_changes

    def test_compare_system_connection_change(self):
        """Test system comparison detects connection status change."""
        updater = StateDeltaUpdater()
        old_system = SystemStatus(
            api_latency=0.05,
            connection_status="CONNECTED",
        )
        new_system = SystemStatus(
            api_latency=0.05,
            connection_status="DISCONNECTED",
        )

        result = updater.compare_system(old_system, new_system)

        assert result.has_changes
        assert "connection_status" in result.changed_fields

    def test_should_update_component_no_changes(self):
        """Test should_update returns False when no changes."""
        updater = StateDeltaUpdater()
        delta = DeltaResult()

        assert not updater.should_update_component("market", delta)

    def test_should_update_component_with_changes(self):
        """Test should_update returns True when there are changes."""
        updater = StateDeltaUpdater()
        delta = DeltaResult()
        delta.add_change("prices.BTC-USD", "50000", "51000")

        assert updater.should_update_component("market", delta)

    def test_float_equal_within_epsilon(self):
        """Test float comparison with epsilon tolerance."""
        updater = StateDeltaUpdater()

        assert updater._float_equal(1.0, 1.0)
        assert updater._float_equal(1.0, 1.0 + 1e-10)  # Within epsilon
        assert not updater._float_equal(1.0, 1.1)  # Beyond epsilon

    def test_decimal_equal_within_epsilon(self):
        """Test decimal comparison with epsilon tolerance."""
        updater = StateDeltaUpdater()

        assert updater._decimal_equal(Decimal("1.0"), Decimal("1.0"))
        assert updater._decimal_equal(Decimal("1.0"), Decimal("1.000000001"))
        assert not updater._decimal_equal(Decimal("1.0"), Decimal("1.1"))

    def test_decimal_equal_handles_none(self):
        """Test decimal comparison handles None values."""
        updater = StateDeltaUpdater()

        assert updater._decimal_equal(None, None)
        assert not updater._decimal_equal(Decimal("1.0"), None)
        assert not updater._decimal_equal(None, Decimal("1.0"))
