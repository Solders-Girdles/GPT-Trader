"""
Tests for resilience metrics and performance snapshots in TuiState.
"""

from __future__ import annotations

from gpt_trader.monitoring.status_reporter import (
    AccountStatus,
    BotStatus,
    EngineStatus,
    HeartbeatStatus,
    MarketStatus,
    PositionStatus,
    RiskStatus,
    StrategyStatus,
    SystemStatus,
)
from gpt_trader.tui.state import TuiState


class TestTuiStateResilience:
    def test_initial_resilience_state(self) -> None:
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

    def test_update_resilience_data_parses_metrics(self) -> None:
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

    def test_update_resilience_data_handles_missing_fields(self) -> None:
        """Test resilience update handles partial/missing data gracefully."""
        state = TuiState()

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

    def test_update_resilience_data_circuit_breaker_all_closed(self) -> None:
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


class TestTuiStatePerformanceSnapshots:
    def test_update_from_bot_status_populates_performance_metrics(self) -> None:
        """Test that update_from_bot_status populates strategy_performance and backtest_performance."""
        state = TuiState()

        strategy_status = StrategyStatus(
            active_strategies=["Momentum"],
            last_decisions=[],
            performance={
                "win_rate": 0.58,
                "profit_factor": 1.65,
                "total_return": 0.082,
                "max_drawdown": -0.041,
                "total_trades": 45,
                "winning_trades": 26,
                "losing_trades": 19,
                "sharpe_ratio": 1.05,
            },
            backtest_performance={
                "win_rate": 0.56,
                "profit_factor": 1.42,
                "total_return": 0.124,
                "max_drawdown": -0.062,
                "total_trades": 120,
                "winning_trades": 67,
                "losing_trades": 53,
            },
        )

        status = BotStatus(
            bot_id="test-bot",
            engine=EngineStatus(),
            market=MarketStatus(),
            positions=PositionStatus(),
            orders=[],
            trades=[],
            account=AccountStatus(),
            strategy=strategy_status,
            risk=RiskStatus(),
            system=SystemStatus(),
            heartbeat=HeartbeatStatus(),
        )

        state.update_from_bot_status(status)

        assert state.strategy_performance.win_rate == 0.58
        assert state.strategy_performance.profit_factor == 1.65
        assert abs(state.strategy_performance.total_return_pct - 8.2) < 0.001
        assert state.strategy_performance.total_trades == 45

        assert state.backtest_performance is not None
        assert state.backtest_performance.win_rate == 0.56
        assert state.backtest_performance.profit_factor == 1.42
        assert state.backtest_performance.total_trades == 120
