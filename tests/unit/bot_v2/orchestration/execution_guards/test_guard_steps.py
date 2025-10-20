"""
Tests for guard step execution, telemetry logging, and exception wrapping.
"""

from unittest.mock import Mock, patch

import pytest

from bot_v2.features.live_trade.guard_errors import (
    RiskGuardComputationError,
    RiskGuardTelemetryError,
)


class TestGuardSteps:
    """Test guard step execution and error handling."""

    @pytest.mark.asyncio
    async def test_run_guard_step_success(self, guard_manager):
        """Test run_guard_step executes successfully."""
        func = Mock()
        guard_manager.run_guard_step("test_guard", func)

        func.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_guard_step_risk_guard_error_recoverable(self, guard_manager):
        """Test run_guard_step handles recoverable RiskGuardError."""
        from bot_v2.features.live_trade.guard_errors import RiskGuardTelemetryError

        error = RiskGuardTelemetryError(guard="test", message="Test error")
        func = Mock(side_effect=error)

        # Should not raise
        guard_manager.run_guard_step("test_guard", func)

        func.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_guard_step_risk_guard_error_non_recoverable(self, guard_manager):
        """Test run_guard_step raises non-recoverable RiskGuardError."""
        from bot_v2.features.live_trade.guard_errors import RiskGuardComputationError

        error = RiskGuardComputationError(guard="test", message="Test error")
        func = Mock(side_effect=error)

        with pytest.raises(RiskGuardComputationError):
            guard_manager.run_guard_step("test_guard", func)

    @pytest.mark.asyncio
    async def test_run_guard_step_generic_exception(self, guard_manager):
        """Test run_guard_step handles generic exceptions."""
        func = Mock(side_effect=ValueError("Test error"))

        with pytest.raises(RiskGuardComputationError) as exc_info:
            guard_manager.run_guard_step("test_guard", func)

        assert exc_info.value.guard == "test_guard"
        assert "Unexpected failure" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_log_guard_telemetry_success(self, guard_manager, runtime_guard_state):
        """Test log_guard_telemetry logs successfully."""
        with patch("bot_v2.orchestration.execution.guards._get_plog") as mock_get_plog:
            mock_plog = Mock()
            mock_get_plog.return_value = mock_plog

            guard_manager.log_guard_telemetry(runtime_guard_state)

            mock_plog.log_pnl.assert_called_once_with(
                symbol="BTC-PERP", realized_pnl=100.0, unrealized_pnl=500.0
            )

    @pytest.mark.asyncio
    async def test_log_guard_telemetry_failure(self, guard_manager, runtime_guard_state):
        """Test log_guard_telemetry handles telemetry failures."""
        with patch("bot_v2.orchestration.execution.guards._get_plog") as mock_get_plog:
            mock_plog = Mock()
            mock_get_plog.return_value = mock_plog
            mock_plog.log_pnl.side_effect = Exception("Telemetry failed")

            with pytest.raises(RiskGuardTelemetryError) as exc_info:
                guard_manager.log_guard_telemetry(runtime_guard_state)

            assert "Failed to emit PnL telemetry" in str(exc_info.value)
            assert len(exc_info.value.details["failures"]) == 1

    @pytest.mark.asyncio
    async def test_run_guards_for_state_success(self, guard_manager, runtime_guard_state):
        """Test run_guards_for_state executes all guards."""
        with patch.object(guard_manager, "run_guard_step") as mock_run_step:
            guard_manager.run_guards_for_state(runtime_guard_state, incremental=False)

            assert mock_run_step.call_count == 7  # All guard steps

    @pytest.mark.asyncio
    async def test_run_runtime_guards_full_run(self, guard_manager, runtime_guard_state):
        """Test run_runtime_guards performs full run."""
        with (
            patch.object(
                guard_manager, "collect_runtime_guard_state", return_value=runtime_guard_state
            ),
            patch.object(guard_manager, "run_guards_for_state") as mock_run_guards,
        ):

            result = guard_manager.run_runtime_guards(force_full=True)

            assert result is runtime_guard_state
            mock_run_guards.assert_called_once_with(runtime_guard_state, False)

    @pytest.mark.asyncio
    async def test_run_runtime_guards_incremental_run(self, guard_manager, runtime_guard_state):
        """Test run_runtime_guards performs incremental run."""
        guard_manager._runtime_guard_state = runtime_guard_state
        guard_manager._runtime_guard_last_full_ts = 1234567890.0 + 30  # Within interval

        with (
            patch.object(guard_manager, "run_guards_for_state") as mock_run_guards,
            patch.object(
                guard_manager, "collect_runtime_guard_state", return_value=runtime_guard_state
            ),
        ):

            result = guard_manager.run_runtime_guards(force_full=False)

            assert result is runtime_guard_state
            mock_run_guards.assert_called_once_with(runtime_guard_state, False)

    @pytest.mark.asyncio
    async def test_run_runtime_guards_forced_full_overrides_incremental(
        self, guard_manager, runtime_guard_state
    ):
        """Test run_runtime_guards forces full run when requested."""
        guard_manager._runtime_guard_state = runtime_guard_state
        guard_manager._runtime_guard_last_full_ts = 1234567890.0 + 30  # Within interval

        with (
            patch.object(
                guard_manager, "collect_runtime_guard_state", return_value=runtime_guard_state
            ),
            patch.object(guard_manager, "run_guards_for_state") as mock_run_guards,
        ):

            result = guard_manager.run_runtime_guards(force_full=True)

            assert result is runtime_guard_state
            mock_run_guards.assert_called_once_with(runtime_guard_state, False)

    @pytest.mark.asyncio
    async def test_should_run_full_guard_dirty_cache(self, guard_manager):
        """Test should_run_full_guard returns True when cache is dirty."""
        guard_manager._runtime_guard_dirty = True

        assert guard_manager.should_run_full_guard(1234567890.0) is True

    @pytest.mark.asyncio
    async def test_should_run_full_guard_no_cached_state(self, guard_manager):
        """Test should_run_full_guard returns True when no cached state."""
        guard_manager._runtime_guard_dirty = False
        guard_manager._runtime_guard_state = None

        assert guard_manager.should_run_full_guard(1234567890.0) is True

    @pytest.mark.asyncio
    async def test_should_run_full_guard_interval_expired(self, guard_manager, runtime_guard_state):
        """Test should_run_full_guard returns True when interval expired."""
        guard_manager._runtime_guard_dirty = False
        guard_manager._runtime_guard_state = runtime_guard_state
        guard_manager._runtime_guard_last_full_ts = 1234567890.0 - 70  # Expired

        assert guard_manager.should_run_full_guard(1234567890.0) is True

    @pytest.mark.asyncio
    async def test_should_run_full_guard_within_interval(self, guard_manager, runtime_guard_state):
        """Test should_run_full_guard returns False when within interval."""
        guard_manager._runtime_guard_dirty = False
        guard_manager._runtime_guard_state = runtime_guard_state
        guard_manager._runtime_guard_last_full_ts = 1234567890.0 - 30  # Within interval

        assert guard_manager.should_run_full_guard(1234567890.0) is False

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, guard_manager, runtime_guard_state):
        """Test invalidate_cache clears state and marks dirty."""
        guard_manager._runtime_guard_state = runtime_guard_state
        guard_manager._runtime_guard_dirty = False

        guard_manager.invalidate_cache()

        assert guard_manager._runtime_guard_state is None
        assert guard_manager._runtime_guard_dirty is True
