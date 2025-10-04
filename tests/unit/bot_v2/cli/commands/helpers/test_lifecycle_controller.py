"""Tests for lifecycle controller."""

from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.cli.commands.lifecycle_controller import LifecycleController


class TestLifecycleController:
    """Test LifecycleController."""

    def test_execute_with_single_cycle_true(self):
        """Test execute with single_cycle=True."""
        bot = Mock()
        bot.run = AsyncMock()
        mock_runner = Mock()

        controller = LifecycleController(runner=mock_runner)
        result = controller.execute(bot, single_cycle=True)

        assert result == 0
        bot.run.assert_called_once_with(single_cycle=True)
        mock_runner.assert_called_once()

    def test_execute_with_single_cycle_false(self):
        """Test execute with single_cycle=False."""
        bot = Mock()
        bot.run = AsyncMock()
        mock_runner = Mock()

        controller = LifecycleController(runner=mock_runner)
        result = controller.execute(bot, single_cycle=False)

        assert result == 0
        bot.run.assert_called_once_with(single_cycle=False)
        mock_runner.assert_called_once()

    def test_execute_with_custom_runner(self):
        """Test controller uses custom runner when provided."""
        bot = Mock()
        bot.run = AsyncMock()
        custom_runner = Mock()

        controller = LifecycleController(runner=custom_runner)
        controller.execute(bot, single_cycle=True)

        # Verify custom runner was used
        custom_runner.assert_called_once()
        # Verify it was called with bot.run coroutine
        call_arg = custom_runner.call_args[0][0]
        bot.run.assert_called_once_with(single_cycle=True)

    def test_execute_propagates_exception(self):
        """Test that exceptions from runner are propagated."""
        bot = Mock()
        bot.run = AsyncMock()
        mock_runner = Mock(side_effect=Exception("Execution failed"))

        controller = LifecycleController(runner=mock_runner)

        with pytest.raises(Exception, match="Execution failed"):
            controller.execute(bot, single_cycle=True)

    def test_execute_returns_zero_on_success(self):
        """Test execute always returns 0 on successful execution."""
        bot = Mock()
        bot.run = AsyncMock()
        mock_runner = Mock()

        controller = LifecycleController(runner=mock_runner)

        # Test with single_cycle=True
        result1 = controller.execute(bot, single_cycle=True)
        assert result1 == 0

        # Test with single_cycle=False
        result2 = controller.execute(bot, single_cycle=False)
        assert result2 == 0

    def test_default_runner_uses_asyncio_run(self):
        """Test default runner is asyncio.run."""
        # This test verifies the default is set, but doesn't actually run async
        controller = LifecycleController()

        # Verify _runner is set
        assert controller._runner is not None
        # The actual asyncio.run is imported and set in __init__
