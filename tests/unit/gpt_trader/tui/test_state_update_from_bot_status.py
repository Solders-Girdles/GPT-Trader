"""
Tests for TuiState.update_from_bot_status.
"""

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.monitoring.status_reporter import (
    AccountStatus,
    BotStatus,
    DecisionEntry,
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


class TestTuiStateUpdateFromBotStatus:
    def test_initial_state(self) -> None:
        state = TuiState()
        assert state.running is False
        assert state.uptime == 0.0
        assert state.market_data.prices == {}
        assert state.position_data.equity == Decimal("0")

    def test_update_from_bot_status(self) -> None:
        state = TuiState()

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

    def test_update_from_bot_status_converts_price_history(self) -> None:
        state = TuiState()

        status = BotStatus(
            market=MarketStatus(
                last_prices={"BTC-USD": Decimal("50000.00")},
                last_price_update=1600000000.0,
                price_history={
                    "BTC-USD": [Decimal("49000.00"), Decimal("50000.00")],
                },
            )
        )

        state.update_from_bot_status(status)

        assert state.market_data.price_history["BTC-USD"] == [
            Decimal("49000.00"),
            Decimal("50000.00"),
        ]

    def test_update_from_bot_status_populates_last_decisions(self) -> None:
        state = TuiState()

        status = BotStatus(
            strategy=StrategyStatus(
                active_strategies=["TrendFollowing"],
                last_decisions=[
                    DecisionEntry(
                        symbol="BTC-USD",
                        action="BUY",
                        reason="Signal",
                        confidence=0.9,
                        indicators={},
                        timestamp=1234567890.0,
                    )
                ],
            )
        )

        state.update_from_bot_status(status)

        assert "BTC-USD" in state.strategy_data.last_decisions
        decision = state.strategy_data.last_decisions["BTC-USD"]
        assert decision.action == "BUY"
        assert decision.confidence == 0.9
        assert decision.timestamp == 1234567890.0

    def test_update_from_bot_status_isolates_component_failures(self) -> None:
        from types import SimpleNamespace

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)

        status = MagicMock()
        status.timestamp = 0.0
        status.observer_interval = 1.0

        status.market = "Not an object"
        status.positions = SimpleNamespace(
            positions={},
            total_unrealized_pnl=Decimal("0"),
            equity=Decimal("10000.00"),
            total_realized_pnl=Decimal("0"),
        )
        status.orders = []
        status.trades = []
        status.account = SimpleNamespace(
            balances=[],
            volume_30d=Decimal("0"),
            fees_30d=Decimal("0"),
            fee_tier="None",
        )
        status.strategy = SimpleNamespace(active_strategies=[], last_decisions=[])
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
            rate_limit_usage="0%",
            memory_usage="0MB",
            cpu_usage="0%",
        )
        status.websocket = SimpleNamespace(
            connected=False,
            last_message_ts=None,
            last_heartbeat_ts=None,
            last_close_ts=None,
            last_error_ts=None,
            gap_count=0,
            reconnect_count=0,
            message_stale=False,
            heartbeat_stale=False,
        )

        state.update_from_bot_status(status)

        assert state.market_data.prices == {}
        assert state.system_data.connection_status == "CONNECTED"

    def test_update_from_bot_status_handles_missing_sections(self) -> None:
        from types import SimpleNamespace

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)

        status = SimpleNamespace(
            timestamp=0.0,
            observer_interval=1.0,
            market=None,
            orders=[],
            trades=[],
            strategy=SimpleNamespace(active_strategies=[]),
            risk=SimpleNamespace(),
            system=SimpleNamespace(
                api_latency=0.0,
                connection_status="CONNECTED",
                rate_limit_usage="0%",
                memory_usage="0MB",
                cpu_usage="0%",
            ),
            websocket=SimpleNamespace(),
        )

        state.update_from_bot_status(status)

        assert state.system_data.connection_status == "CONNECTED"
        assert state.account_data is not None
        assert state.position_data is not None

    def test_reactive_updates(self) -> None:
        state = TuiState()
        state.running = True
        assert state.running is True
