from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpt_trader.monitoring.status_reporter import (
    AccountStatus,
    BalanceEntry,
    BotStatus,
    DecisionEntry,
    GuardStatus,
    MarketStatus,
    OrderStatus,
    PositionStatus,
    RiskStatus,
    StrategyStatus,
    SystemStatus,
    TradeStatus,
)
from gpt_trader.tui.state import TuiState


def make_status(**overrides):
    """Create a complete BotStatus with optional field overrides."""
    status = BotStatus()
    for key, value in overrides.items():
        setattr(status, key, value)
    return status


@pytest.fixture
def tui_state():
    return TuiState()


def test_update_market_data(tui_state):
    # MarketStatus: object with attributes, internal collections are dicts
    status = make_status(
        market=MarketStatus(
            last_prices={"BTC-USD": Decimal("50000.00")},
            last_price_update=1234567890.0,
            price_history={"BTC-USD": [Decimal("49000.00"), Decimal("50000.00")]},
        )
    )
    tui_state.update_from_bot_status(status)
    assert tui_state.market_data.prices == {"BTC-USD": Decimal("50000.00")}
    assert tui_state.market_data.last_update == 1234567890.0


def test_update_position_data_dict(tui_state):
    # PositionStatus: object with .positions (dict of dicts)
    status = make_status(
        positions=PositionStatus(
            positions={
                "BTC-USD": {
                    "quantity": "1.0",
                    "entry_price": "45000.00",
                    "unrealized_pnl": "5000.00",
                    "mark_price": "50000.00",
                    "side": "LONG",
                }
            },
            total_unrealized_pnl=Decimal("5000.00"),
            equity=Decimal("100000.00"),
        )
    )
    tui_state.update_from_bot_status(status)
    assert "BTC-USD" in tui_state.position_data.positions
    pos = tui_state.position_data.positions["BTC-USD"]
    assert pos.quantity == Decimal("1.0")
    assert pos.entry_price == Decimal("45000.00")
    assert tui_state.position_data.total_unrealized_pnl == Decimal("5000.00")
    assert tui_state.position_data.equity == Decimal("100000.00")


def test_update_order_data(tui_state):
    # list[OrderStatus] -> list of objects
    order_obj = OrderStatus(
        order_id="1",
        symbol="BTC-USD",
        side="BUY",
        quantity=Decimal("0.5"),
        price=Decimal("49000.00"),
        status="OPEN",
        order_type="LIMIT",
        time_in_force="GTC",
        creation_time=1672574400.0,
    )
    status = make_status(orders=[order_obj])
    tui_state.update_from_bot_status(status)
    assert len(tui_state.order_data.orders) == 1
    order = tui_state.order_data.orders[0]
    assert order.order_id == "1"
    assert order.symbol == "BTC-USD"
    # Verify Decimal conversion if accessed
    assert order.quantity == Decimal("0.5")


def test_update_trade_data(tui_state):
    # list[TradeStatus] -> list of objects
    trade_obj = TradeStatus(
        trade_id="t1",
        symbol="BTC-USD",
        side="SELL",
        quantity=Decimal("0.1"),
        price=Decimal("51000.00"),
        order_id="o1",
        time="2023-01-01T12:00:00Z",
        fee=Decimal("5.00"),
    )
    status = make_status(trades=[trade_obj])
    tui_state.update_from_bot_status(status)
    assert len(tui_state.trade_data.trades) == 1
    trade = tui_state.trade_data.trades[0]
    assert trade.trade_id == "t1"
    assert trade.symbol == "BTC-USD"
    assert trade.price == Decimal("51000.00")


def test_update_account_data(tui_state):
    # AccountStatus: object with .balances (list of objects)
    balance_obj = BalanceEntry(
        asset="USD",
        total=Decimal("50000.00"),
        available=Decimal("40000.00"),
        hold=Decimal("10000.00"),
    )
    status = make_status(
        account=AccountStatus(
            volume_30d=Decimal("1000000.00"),
            fees_30d=Decimal("50.00"),
            fee_tier="Taker",
            balances=[balance_obj],
        )
    )
    tui_state.update_from_bot_status(status)
    assert tui_state.account_data.volume_30d == Decimal("1000000.00")
    assert len(tui_state.account_data.balances) == 1
    assert tui_state.account_data.balances[0].asset == "USD"
    assert tui_state.account_data.balances[0].total == Decimal("50000.00")


def test_update_strategy_data(tui_state):
    # StrategyStatus: object with .last_decisions (list of DecisionEntry objects)
    decision_obj = DecisionEntry(
        symbol="BTC-USD",
        action="BUY",
        reason="Signal",
        confidence=0.9,
        indicators={},
        timestamp=1234567890.0,
    )
    status = make_status(
        strategy=StrategyStatus(
            active_strategies=["TrendFollowing"],
            last_decisions=[decision_obj],
        )
    )
    tui_state.update_from_bot_status(status)
    assert "BTC-USD" in tui_state.strategy_data.last_decisions
    decision = tui_state.strategy_data.last_decisions["BTC-USD"]
    assert decision.action == "BUY"
    assert decision.confidence == 0.9


def test_update_risk_data(tui_state):
    # RiskStatus: object
    status = make_status(
        risk=RiskStatus(
            max_leverage=5.0,
            daily_loss_limit_pct=0.02,
            current_daily_loss_pct=0.01,
            reduce_only_mode=True,
            reduce_only_reason="Drawdown",
            guards=[GuardStatus(name="DrawdownGuard")],
        )
    )
    tui_state.update_from_bot_status(status)
    assert tui_state.risk_data.max_leverage == 5.0
    assert tui_state.risk_data.reduce_only_mode is True


def test_update_system_data(tui_state):
    # SystemStatus: object
    status = make_status(
        system=SystemStatus(
            api_latency=0.05,
            connection_status="CONNECTED",
            rate_limit_usage="10%",
            memory_usage="500MB",
            cpu_usage="5%",
        )
    )
    tui_state.update_from_bot_status(status)
    assert tui_state.system_data.connection_status == "CONNECTED"


def test_malformed_data_handling(tui_state):
    # Test that malformed data in one section doesn't crash the whole update
    status = MagicMock()
    # Mocking behavior: market is a string (will raise AttributeError on access to attributes if TuiState tries)
    # Actually TuiState accesses status.market, passes to _update_market_data.
    # _update_market_data attempts to read attributes.
    status.market = "Not an object"

    # FIX: Define observer_interval as a float so > 0 comparison works
    status.observer_interval = 1.0

    # ensure other fields exist to avoid errors in update loop before component update
    # Need full mock objects for these to pass Update methods

    # Positions
    status.positions = SimpleNamespace(positions={}, total_unrealized_pnl="0.0", equity="10000.0")
    status.orders = []
    status.trades = []

    # Account
    status.account = SimpleNamespace(balances=[], volume_30d="0.0", fees_30d="0.0", fee_tier="None")

    # Strategy
    status.strategy = SimpleNamespace(last_decisions=[], active_strategies=[])

    # Risk
    status.risk = SimpleNamespace(
        max_leverage=1.0,
        daily_loss_limit_pct=0.01,
        current_daily_loss_pct=0.0,
        reduce_only_mode=False,
        reduce_only_reason="",
        guards=[],
    )

    status.system = SimpleNamespace(
        api_latency=0.0,
        connection_status="CONNECTED",
        rate_limit_usage=0,
        memory_usage=0,
        cpu_usage=0,
    )

    tui_state.update_from_bot_status(status)

    # Market data should remain default or unchanged
    assert tui_state.market_data.prices == {}

    # System data should have updated despite other failures
    assert tui_state.system_data.connection_status == "CONNECTED"


def test_partial_update(tui_state):
    # Test updating where some main components are missing on status object
    # TuiState catches failures.
    status = SimpleNamespace(
        system=SimpleNamespace(
            api_latency=0.0,
            connection_status="CONNECTED",
            rate_limit_usage=0,
            memory_usage=0,
            cpu_usage=0,
        )
    )
    # Other fields missing. Loop will try status.market -> raises AttributeError -> caught.

    tui_state.update_from_bot_status(status)

    assert tui_state.system_data.connection_status == "CONNECTED"
    assert tui_state.market_data.prices == {}
