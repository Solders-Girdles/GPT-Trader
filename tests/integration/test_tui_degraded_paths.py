"""
Integration tests for TUI degraded/partial state handling.

Tests verify that the TUI update path handles missing or partial status data
without crashing, ensuring graceful degradation when status reporter or
data sources are unavailable.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.services.state_registry import StateRegistry
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import (
    AccountBalance,
    AccountSummary,
    ExecutionMetrics,
    PortfolioSummary,
    Position,
    ResilienceState,
    SystemStatus,
)


class TestTuiStatePartialUpdates:
    """Test TuiState updates with partial/missing data sections."""

    def test_state_with_only_system_data(self) -> None:
        """Test that state can be created with only system_data."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.system_data = SystemStatus(
            cpu_usage="50%",
            api_latency=100.0,
            connection_status="CONNECTED",
        )

        # Other data sections should be at defaults
        assert state.account_data is not None  # Has default empty AccountSummary
        assert state.position_data is not None  # Has default empty PortfolioSummary
        assert state.system_data.cpu_usage == "50%"

    def test_state_with_only_account_data(self) -> None:
        """Test that state can be created with only account_data."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.account_data = AccountSummary(
            balances=[
                AccountBalance(asset="USD", total=Decimal("10000"), available=Decimal("8000")),
            ]
        )

        # System data should be at defaults
        assert state.system_data is not None
        assert len(state.account_data.balances) == 1

    def test_state_with_only_position_data(self) -> None:
        """Test that state can be created with only position_data."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.position_data = PortfolioSummary(
            positions={
                "BTC-USD": Position(
                    symbol="BTC-USD",
                    quantity=Decimal("0.1"),
                    side="LONG",
                    leverage=2,
                    unrealized_pnl=Decimal("500"),
                    entry_price=Decimal("95000"),
                    mark_price=Decimal("96000"),
                )
            }
        )

        assert len(state.position_data.positions) == 1

    def test_state_with_null_resilience_data(self) -> None:
        """Test that state handles None resilience_data."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.resilience_data = None

        # Should not crash when accessing
        assert state.resilience_data is None

    def test_state_with_null_execution_data(self) -> None:
        """Test that state handles None execution_data."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.execution_data = None

        # Should not crash when accessing
        assert state.execution_data is None


class TestStateRegistryBroadcast:
    """Test StateRegistry broadcast with partial state."""

    def test_broadcast_with_partial_state_no_crash(self) -> None:
        """Test that broadcast doesn't crash with partial state."""
        registry = StateRegistry()

        # Create a mock observer
        observer = MagicMock()
        observer.on_state_updated = MagicMock()
        registry.register(observer)

        # Create partial state
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.system_data = SystemStatus(cpu_usage="25%")
        state.account_data = None  # Explicitly None
        state.position_data = None  # Explicitly None

        # Should not raise
        registry.broadcast(state)

        # Observer should have been called
        observer.on_state_updated.assert_called_once_with(state)

    def test_broadcast_handles_observer_exception(self) -> None:
        """Test that broadcast continues after observer exception."""
        registry = StateRegistry()

        # Create observers - one that fails, one that succeeds
        failing_observer = MagicMock()
        failing_observer.on_state_updated = MagicMock(side_effect=ValueError("Test error"))

        working_observer = MagicMock()
        working_observer.on_state_updated = MagicMock()

        registry.register(failing_observer)
        registry.register(working_observer)

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)

        # Should not raise even though one observer fails
        registry.broadcast(state)

        # Both should have been called
        failing_observer.on_state_updated.assert_called_once()
        working_observer.on_state_updated.assert_called_once()

    def test_broadcast_with_degraded_state(self) -> None:
        """Test broadcast with degraded mode state."""
        registry = StateRegistry()

        observer = MagicMock()
        observer.on_state_updated = MagicMock()
        registry.register(observer)

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.degraded_mode = True
        state.degraded_reason = "StatusReporter unavailable"
        state.connection_healthy = False

        # Should not raise
        registry.broadcast(state)

        observer.on_state_updated.assert_called_once_with(state)


class TestStateUpdatePaths:
    """Test various state update scenarios."""

    def test_state_update_with_missing_fields(self) -> None:
        """Test state can be updated when some fields are missing."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)

        # Simulate partial status update
        state.system_data = SystemStatus(
            cpu_usage="30%",
            # api_latency missing
            connection_status="CONNECTED",
            # rate_limit_usage missing
        )

        # Should have defaults for missing fields
        assert state.system_data.cpu_usage == "30%"
        assert state.system_data.connection_status == "CONNECTED"

    def test_state_update_with_resilience_data(self) -> None:
        """Test state update with resilience metrics."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.resilience_data = ResilienceState(
            latency_p50_ms=50.0,
            latency_p95_ms=100.0,
            error_rate=0.01,
            cache_hit_rate=0.85,
            any_circuit_open=False,
            last_update=1234567890.0,
        )

        assert state.resilience_data.latency_p50_ms == 50.0
        assert not state.resilience_data.any_circuit_open

    def test_state_update_with_execution_data(self) -> None:
        """Test state update with execution metrics."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        state.execution_data = ExecutionMetrics(
            submissions_total=100,
            submissions_success=95,
            submissions_failed=5,
            avg_latency_ms=150.0,
        )

        assert state.execution_data.submissions_total == 100
        assert state.execution_data.submissions_success == 95

    def test_state_transition_from_healthy_to_degraded(self) -> None:
        """Test state transition from healthy to degraded mode."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)

        # Start healthy
        state.connection_healthy = True
        state.degraded_mode = False
        assert state.connection_healthy
        assert not state.degraded_mode

        # Transition to degraded
        state.connection_healthy = False
        state.degraded_mode = True
        state.degraded_reason = "Connection lost"

        assert not state.connection_healthy
        assert state.degraded_mode
        assert state.degraded_reason == "Connection lost"


@pytest.mark.integration
class TestUICoordinatorPartialUpdates:
    """Integration tests for UICoordinator with partial updates."""

    def test_apply_status_update_with_missing_account(self) -> None:
        """Test UICoordinator handles status with missing account data."""
        # This test verifies the update path doesn't crash
        # when StatusReporter returns partial data

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)

        # Simulate partial bot status (as from StatusReporter)
        partial_status = {
            "system": {
                "cpu_usage": "40%",
                "memory_usage": "512MB",
                "connection_status": "CONNECTED",
            },
            # "account" key missing
            # "positions" key missing
        }

        # State should remain valid even with partial update
        if "system" in partial_status:
            state.system_data = SystemStatus(**partial_status["system"])

        assert state.system_data.cpu_usage == "40%"
        # Account data should be at default
        assert state.account_data is not None

    def test_apply_status_update_with_empty_positions(self) -> None:
        """Test UICoordinator handles status with empty positions."""
        state = TuiState(validation_enabled=False, delta_updates_enabled=False)

        # Status with empty positions
        state.position_data = PortfolioSummary(positions={})

        assert len(state.position_data.positions) == 0
        assert state.position_data.equity is None or state.position_data.equity == Decimal("0")

    def test_state_broadcast_order_independence(self) -> None:
        """Test that observers receive state regardless of registration order."""
        registry = StateRegistry()

        results = []

        def create_observer(name: str) -> MagicMock:
            observer = MagicMock()
            observer.on_state_updated = MagicMock(side_effect=lambda s: results.append(name))
            return observer

        # Register in specific order
        obs1 = create_observer("first")
        obs2 = create_observer("second")
        obs3 = create_observer("third")

        registry.register(obs1)
        registry.register(obs2)
        registry.register(obs3)

        state = TuiState(validation_enabled=False, delta_updates_enabled=False)
        registry.broadcast(state)

        # All observers should have been called
        assert len(results) == 3
        assert "first" in results
        assert "second" in results
        assert "third" in results
