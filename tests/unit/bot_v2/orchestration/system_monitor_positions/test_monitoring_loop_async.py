"""Tests for PositionReconciler async monitoring loop functionality."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bot_v2.orchestration.system_monitor_positions import PositionReconciler


@pytest.mark.asyncio
class TestMonitoringLoop:
    """Test async monitoring loop behavior."""

    async def test_run_initializes_last_positions(
        self, reconciler: PositionReconciler, fake_bot, sample_positions, reset_iteration_counter
    ) -> None:
        """Test run initializes last_positions when empty."""
        # Setup: empty last_positions, one fake position returned
        fake_bot.runtime_state.last_positions = {}
        fake_bot.broker.list_positions.return_value = sample_positions

        # Stop after 1 iteration
        fake_bot.stop_after_iterations(1)

        await reconciler.run(fake_bot, interval_seconds=1)

        # Verify state updated with normalized positions
        expected = {
            "BTC-PERP": {"quantity": "0.5", "side": "long"},
            "ETH-PERP": {"quantity": "1.0", "side": "short"},
        }
        assert fake_bot.runtime_state.last_positions == expected

    async def test_run_emits_diff_and_updates_state(
        self, reconciler: PositionReconciler, fake_bot, reset_iteration_counter
    ) -> None:
        """Test run detects changes and updates state accordingly."""
        # Setup: seed last_positions with different data
        fake_bot.runtime_state.last_positions = {
            "BTC-PERP": {"quantity": "0.3", "side": "long"},
            "ETH-PERP": {"quantity": "1.0", "side": "short"},
        }

        # Current positions with change in BTC quantity
        current_positions = [
            SimpleNamespace(symbol="BTC-PERP", quantity=Decimal("0.5"), side="long"),
        ]
        fake_bot.broker.list_positions.return_value = current_positions

        # Mock the change emission method
        reconciler._emit_position_changes = MagicMock()

        # Stop after 1 iteration
        fake_bot.stop_after_iterations(1)

        await reconciler.run(fake_bot, interval_seconds=1)

        # Verify emission was called
        reconciler._emit_position_changes.assert_called_once()

        # Verify state updated with new positions
        expected = {"BTC-PERP": {"quantity": "0.5", "side": "long"}}
        assert fake_bot.runtime_state.last_positions == expected

    async def test_run_handles_exception(
        self, reconciler: PositionReconciler, fake_bot, caplog, reset_iteration_counter
    ) -> None:
        """Test run handles exceptions gracefully and continues running."""
        # Make _normalize_positions raise an exception
        with pytest.MonkeyPatch().context() as m:

            def normalize_that_raises(positions):
                raise RuntimeError("Test exception")

            m.setattr(reconciler, "_normalize_positions", normalize_that_raises)

            # Stop after 1 iteration
            fake_bot.stop_after_iterations(1)

            await reconciler.run(fake_bot, interval_seconds=1)

            # Verify debug log captured
            assert "Position reconciliation error" in caplog.text
            assert "Test exception" in caplog.text

            # Verify bot is still considered running (exception didn't crash loop)
            assert fake_bot.running is False  # False due to our stop condition

    async def test_run_uses_interval_sleep(
        self, reconciler: PositionReconciler, fake_bot, reset_iteration_counter
    ) -> None:
        """Test run respects configured interval for sleep."""
        # Setup empty positions to avoid state changes
        fake_bot.broker.list_positions.return_value = []
        fake_bot.runtime_state.last_positions = {}

        # Stop after 2 iterations to test sleep calls
        fake_bot.stop_after_iterations(2)

        with pytest.MonkeyPatch().context() as m:
            sleep_spy = MagicMock()
            m.setattr("bot_v2.orchestration.system_monitor_positions.asyncio.sleep", sleep_spy)

            await reconciler.run(fake_bot, interval_seconds=5)

            # Verify sleep was called with correct interval
            sleep_spy.assert_called_with(5)

    async def test_run_stops_when_bot_not_running(
        self, reconciler: PositionReconciler, fake_bot, reset_iteration_counter
    ) -> None:
        """Test run exits immediately when bot.running is False."""
        fake_bot.running = False

        await reconciler.run(fake_bot, interval_seconds=1)

        # Should not have made any broker calls
        fake_bot.broker.list_positions.assert_not_called()

    async def test_run_with_bot_running_false_after_exception(
        self, reconciler: PositionReconciler, fake_bot, reset_iteration_counter
    ) -> None:
        """Test run handles case where bot.running becomes False during execution."""
        # Start with bot running
        fake_bot.running = True

        # Make broker call set running to False (simulating external stop)
        def stop_bot(*args, **kwargs):
            fake_bot.running = False
            return []

        fake_bot.broker.list_positions.side_effect = stop_bot

        await reconciler.run(fake_bot, interval_seconds=1)

        # Should have made one attempt before stopping
        fake_bot.broker.list_positions.assert_called_once()

    async def test_run_multiple_iterations(
        self, reconciler: PositionReconciler, fake_bot, reset_iteration_counter
    ) -> None:
        """Test run processes multiple iterations correctly."""
        # Setup positions that change each iteration
        positions_by_iteration = [
            [],  # First iteration: empty
            [
                SimpleNamespace(symbol="BTC-PERP", quantity=Decimal("0.5"), side="long")
            ],  # Second iteration
        ]

        iteration_count = {"calls": 0}

        def positions_side_effect(*args, **kwargs):
            result = positions_by_iteration[
                min(iteration_count["calls"], len(positions_by_iteration) - 1)
            ]
            iteration_count["calls"] += 1
            return result

        fake_bot.broker.list_positions.side_effect = positions_side_effect

        # Stop after 2 iterations
        fake_bot.stop_after_iterations(2)

        await reconciler.run(fake_bot, interval_seconds=1)

        # Verify broker called multiple times
        assert fake_bot.broker.list_positions.call_count >= 2
        assert iteration_count["calls"] >= 2

        # Final state should have BTC position
        expected = {"BTC-PERP": {"quantity": "0.5", "side": "long"}}
        assert fake_bot.runtime_state.last_positions == expected
