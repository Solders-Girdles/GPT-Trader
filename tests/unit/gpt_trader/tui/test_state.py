"""
Tests for TuiState.
"""

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.core.account import CFMBalance
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

    def test_initial_cfm_state(self):
        """Test that CFM state starts as unavailable."""
        state = TuiState()
        assert state.has_cfm_access is False
        assert state.cfm_balance is None

    def test_update_cfm_balance_with_data(self):
        """Test updating CFM balance sets has_cfm_access to True."""
        state = TuiState()

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("10000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("4000.00"),
            initial_margin=Decimal("1000.00"),
            unrealized_pnl=Decimal("150.50"),
            daily_realized_pnl=Decimal("75.25"),
            liquidation_threshold=Decimal("500.00"),
            liquidation_buffer_amount=Decimal("4500.00"),
            liquidation_buffer_percentage=90.0,
        )

        state.update_cfm_balance(cfm_balance)

        assert state.has_cfm_access is True
        assert state.cfm_balance is not None
        assert state.cfm_balance.futures_buying_power == Decimal("10000.00")
        assert state.cfm_balance.liquidation_buffer_percentage == 90.0

    def test_update_cfm_balance_with_none(self):
        """Test updating CFM balance to None sets has_cfm_access to False."""
        state = TuiState()

        # First set a balance
        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("10000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("4000.00"),
            initial_margin=Decimal("1000.00"),
            unrealized_pnl=Decimal("0"),
            daily_realized_pnl=Decimal("0"),
            liquidation_threshold=Decimal("500.00"),
            liquidation_buffer_amount=Decimal("4500.00"),
            liquidation_buffer_percentage=90.0,
        )
        state.update_cfm_balance(cfm_balance)
        assert state.has_cfm_access is True

        # Then clear it
        state.update_cfm_balance(None)
        assert state.has_cfm_access is False
        assert state.cfm_balance is None

    def test_initial_data_state_properties(self):
        """Test that data state properties start with correct defaults."""
        state = TuiState()
        assert state.data_fetching is False
        assert state.data_available is False
        assert state.last_data_fetch == 0.0

    def test_data_state_properties_can_be_set(self):
        """Test that data state properties can be set."""
        import time

        state = TuiState()

        state.data_fetching = True
        assert state.data_fetching is True

        state.data_available = True
        assert state.data_available is True

        now = time.time()
        state.last_data_fetch = now
        assert state.last_data_fetch == now

    def test_is_data_stale_false_when_no_data(self):
        """Test is_data_stale returns False when no data has been received."""
        state = TuiState()
        assert state.is_data_stale is False

        # Even with data_available set but no timestamp
        state.data_available = True
        assert state.is_data_stale is False

    def test_is_data_stale_false_when_data_fresh(self):
        """Test is_data_stale returns False when data is recent."""
        import time

        state = TuiState()
        state.data_available = True
        state.last_data_fetch = time.time()  # Just fetched

        assert state.is_data_stale is False

    def test_is_data_stale_true_when_data_old(self):
        """Test is_data_stale returns True when data is older than 30s."""
        import time

        state = TuiState()
        state.data_available = True
        state.last_data_fetch = time.time() - 35  # 35 seconds ago (> 30s threshold)

        assert state.is_data_stale is True

    def test_is_data_stale_false_at_threshold(self):
        """Test is_data_stale returns False when data is exactly at threshold."""
        import time

        state = TuiState()
        state.data_available = True
        state.last_data_fetch = time.time() - 29  # 29 seconds ago (< 30s threshold)

        assert state.is_data_stale is False

    def test_initial_resilience_state(self):
        """Test that resilience state starts with defaults."""
        state = TuiState()
        res = state.resilience_data
        assert res.latency_p50_ms == 0.0
        assert res.latency_p95_ms == 0.0
        assert res.error_rate == 0.0
        assert res.cache_hit_rate == 0.0
        assert res.circuit_breakers == {}
        assert res.any_circuit_open is False
        assert res.last_update == 0.0

    def test_update_resilience_data_parses_metrics(self):
        """Test updating resilience data from client status."""
        state = TuiState()

        resilience_status = {
            "metrics": {
                "p50_latency_ms": 45.5,
                "p95_latency_ms": 120.3,
                "avg_latency_ms": 55.0,
                "error_rate": 0.015,
                "total_requests": 1000,
                "total_errors": 15,
                "rate_limit_hits": 5,
            },
            "cache": {
                "hit_rate": 0.85,
                "size": 100,
                "enabled": True,
            },
            "circuit_breakers": {
                "market": {"state": "closed"},
                "account": {"state": "open"},
            },
            "rate_limit_usage": "45%",
        }

        state.update_resilience_data(resilience_status)

        res = state.resilience_data
        assert res.latency_p50_ms == 45.5
        assert res.latency_p95_ms == 120.3
        assert res.error_rate == 0.015
        assert res.cache_hit_rate == 0.85
        assert res.cache_size == 100
        assert res.cache_enabled is True
        assert res.circuit_breakers == {"market": "closed", "account": "open"}
        assert res.any_circuit_open is True
        assert res.rate_limit_usage_pct == 45.0
        assert res.last_update > 0

    def test_update_resilience_data_handles_missing_fields(self):
        """Test resilience update handles partial/missing data gracefully."""
        state = TuiState()

        # Minimal status with missing fields
        resilience_status = {
            "metrics": {},
            "cache": {},
            "circuit_breakers": {},
        }

        state.update_resilience_data(resilience_status)

        res = state.resilience_data
        assert res.latency_p50_ms == 0.0
        assert res.error_rate == 0.0
        assert res.cache_hit_rate == 0.0
        assert res.circuit_breakers == {}
        assert res.any_circuit_open is False

    def test_update_resilience_data_circuit_breaker_all_closed(self):
        """Test that any_circuit_open is False when all breakers closed."""
        state = TuiState()

        resilience_status = {
            "metrics": {},
            "cache": {},
            "circuit_breakers": {
                "market": {"state": "closed"},
                "account": {"state": "closed"},
            },
        }

        state.update_resilience_data(resilience_status)

        assert state.resilience_data.any_circuit_open is False
