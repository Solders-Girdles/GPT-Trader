import pytest

from gpt_trader.tui.state import TuiState


@pytest.fixture
def tui_state():
    return TuiState()


def test_update_market_data(tui_state):
    status = {
        "market": {
            "last_prices": {"BTC-USD": "50000.00"},
            "last_price_update": 1234567890.0,
            "price_history": {"BTC-USD": ["49000.00", "50000.00"]},
        }
    }
    tui_state.update_from_bot_status(status)
    assert tui_state.market_data.prices == {"BTC-USD": "50000.00"}
    assert tui_state.market_data.last_update == 1234567890.0


def test_update_position_data_dict(tui_state):
    status = {
        "positions": {
            "BTC-USD": {
                "quantity": "1.0",
                "entry_price": "45000.00",
                "unrealized_pnl": "5000.00",
                "mark_price": "50000.00",
                "side": "LONG",
            },
            "total_unrealized_pnl": "5000.00",
            "equity": "100000.00",
        }
    }
    tui_state.update_from_bot_status(status)
    assert "BTC-USD" in tui_state.position_data.positions
    pos = tui_state.position_data.positions["BTC-USD"]
    assert pos.quantity == "1.0"
    assert pos.entry_price == "45000.00"
    assert tui_state.position_data.total_unrealized_pnl == "5000.00"
    assert tui_state.position_data.equity == "100000.00"


def test_update_order_data(tui_state):
    status = {
        "orders": [
            {
                "order_id": "1",
                "symbol": "BTC-USD",
                "side": "BUY",
                "quantity": "0.5",
                "price": "49000.00",
                "status": "OPEN",
            }
        ]
    }
    tui_state.update_from_bot_status(status)
    assert len(tui_state.order_data.orders) == 1
    order = tui_state.order_data.orders[0]
    assert order.order_id == "1"
    assert order.symbol == "BTC-USD"


def test_update_trade_data(tui_state):
    status = {
        "trades": [
            {
                "trade_id": "t1",
                "product_id": "BTC-USD",
                "side": "SELL",
                "quantity": "0.1",
                "price": "51000.00",
                "order_id": "o1",
                "time": "2023-01-01T12:00:00Z",
            }
        ]
    }
    tui_state.update_from_bot_status(status)
    assert len(tui_state.trade_data.trades) == 1
    trade = tui_state.trade_data.trades[0]
    assert trade.trade_id == "t1"
    assert trade.symbol == "BTC-USD"


def test_update_account_data(tui_state):
    status = {
        "account": {
            "volume_30d": "1000000.00",
            "fees_30d": "50.00",
            "fee_tier": "Taker",
            "balances": [{"asset": "USD", "total": "50000.00", "available": "40000.00"}],
        }
    }
    tui_state.update_from_bot_status(status)
    assert tui_state.account_data.volume_30d == "1000000.00"
    assert len(tui_state.account_data.balances) == 1
    assert tui_state.account_data.balances[0].asset == "USD"


def test_update_strategy_data(tui_state):
    status = {
        "strategy": {
            "active_strategies": ["TrendFollowing"],
            "last_decisions": [
                {
                    "symbol": "BTC-USD",
                    "action": "BUY",
                    "reason": "Signal",
                    "confidence": 0.9,
                    "timestamp": 1234567890.0,
                }
            ],
        }
    }
    tui_state.update_from_bot_status(status)
    assert "BTC-USD" in tui_state.strategy_data.last_decisions
    decision = tui_state.strategy_data.last_decisions["BTC-USD"]
    assert decision.action == "BUY"
    assert decision.confidence == 0.9


def test_update_risk_data(tui_state):
    status = {
        "risk": {
            "max_leverage": 5.0,
            "daily_loss_limit_pct": 0.02,
            "current_daily_loss_pct": 0.01,
            "reduce_only_mode": True,
            "active_guards": ["DrawdownGuard"],
        }
    }
    tui_state.update_from_bot_status(status)
    assert tui_state.risk_data.max_leverage == 5.0
    assert tui_state.risk_data.reduce_only_mode is True


def test_update_system_data(tui_state):
    status = {
        "system": {
            "api_latency": 0.05,
            "connection_status": "CONNECTED",
            "rate_limit_usage": "10%",
            "memory_usage": "500MB",
            "cpu_usage": "5%",
        }
    }
    tui_state.update_from_bot_status(status)
    assert tui_state.system_data.connection_status == "CONNECTED"


def test_malformed_data_handling(tui_state):
    # Test that malformed data in one section doesn't crash the whole update
    status = {
        "market": "Not a dict",  # Should be handled gracefully
        "positions": {"BTC-USD": "Not a dict"},  # Should be handled gracefully
        "system": {"connection_status": "CONNECTED"},  # Should still update
    }
    tui_state.update_from_bot_status(status)

    # Market data should remain default or unchanged (empty in this new instance)
    assert tui_state.market_data.prices == {}

    # System data should have updated despite other failures
    assert tui_state.system_data.connection_status == "CONNECTED"


def test_partial_update(tui_state):
    # Test updating only some fields
    status = {"system": {"connection_status": "CONNECTED"}}
    tui_state.update_from_bot_status(status)
    assert tui_state.system_data.connection_status == "CONNECTED"
    # Other fields should remain default
    assert tui_state.market_data.prices == {}
