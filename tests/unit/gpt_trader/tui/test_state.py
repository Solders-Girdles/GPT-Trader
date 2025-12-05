"""
Tests for TuiState.
"""

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.monitoring.status_reporter import (
    AccountStatus,
    BotStatus,
    EngineStatus,
    HeartbeatStatus,
    MarketStatus,
    OrderStatus,
    PositionStatus,
    RiskStatus,
    StrategyStatus,
    SystemStatus,
    TradeStatus,
)
from gpt_trader.tui.state import TuiState


class TestTuiState:
    def test_initial_state(self):
        state = TuiState()
        assert state.running is False
        assert state.uptime == 0.0
        assert state.market_data.prices == {}
        assert state.position_data.equity == Decimal("0")

    def test_update_from_bot_status(self):
        state = TuiState()

        # Create typed BotStatus with Decimal values
        status = BotStatus(
            bot_id="test-bot",
            timestamp=1600000000.0,
            timestamp_iso="2020-09-13T12:26:40Z",
            version="test",
            engine=EngineStatus(),
            market=MarketStatus(
                symbols=["BTC-USD"],
                last_prices={"BTC-USD": Decimal("50000.00")},
                last_price_update=1600000000.0,
            ),
            positions=PositionStatus(
                count=1,
                symbols=["BTC-USD"],
                equity=Decimal("10000.00"),
                total_unrealized_pnl=Decimal("500.00"),
                positions={"BTC-USD": {"quantity": Decimal("0.1")}},
            ),
            orders=[
                OrderStatus(
                    order_id="1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("0.1"),
                    price=Decimal("50000.00"),
                    status="OPEN",
                )
            ],
            trades=[
                TradeStatus(
                    trade_id="t1",
                    symbol="BTC-USD",
                    side="BUY",
                    quantity=Decimal("0.05"),
                    price=Decimal("49990.00"),
                    time="2020-09-13T12:26:45Z",
                    order_id="1",
                )
            ],
            account=AccountStatus(),
            strategy=StrategyStatus(),
            risk=RiskStatus(),
            system=SystemStatus(),
            heartbeat=HeartbeatStatus(),
        )

        mock_runtime = MagicMock()
        mock_runtime.uptime = 120.0

        state.update_from_bot_status(status, mock_runtime)

        assert state.market_data.prices["BTC-USD"] == Decimal("50000.00")
        assert state.market_data.last_update == 1600000000.0
        assert state.position_data.equity == Decimal("10000.00")
        assert state.position_data.total_unrealized_pnl == Decimal("500.00")
        assert state.uptime == 120.0

    def test_reactive_updates(self):
        # This tests that setting properties works as expected for a reactive widget
        state = TuiState()
        state.running = True
        assert state.running is True
