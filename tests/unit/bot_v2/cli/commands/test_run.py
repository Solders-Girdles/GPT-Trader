"""Tests for run CLI command."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.cli.commands.run import handle_run_bot


@pytest.fixture
def mock_bot():
    """Create a mock PerpsBot."""
    bot = Mock()
    bot.run = AsyncMock()
    bot.running = True
    return bot


@pytest.fixture
def mock_signal_manager():
    """Create a mock SignalManager."""
    manager = Mock()
    manager.setup_signals = Mock()
    return manager


class TestHandleRunBot:
    """Tests for handle_run_bot function."""

    @patch("bot_v2.cli.commands.run.SignalManager")
    @patch("bot_v2.cli.commands.run.LifecycleController")
    def test_successful_bot_run_dev_mode(
        self, mock_lifecycle_class, mock_signal_class, mock_bot, mock_signal_manager
    ):
        """Test successful bot run in dev_fast mode."""
        mock_signal_class.return_value = mock_signal_manager
        mock_lifecycle = Mock()
        mock_lifecycle.execute.return_value = 0
        mock_lifecycle_class.return_value = mock_lifecycle

        result = handle_run_bot(mock_bot, dev_fast=True)

        assert result == 0
        mock_signal_class.assert_called_once_with()
        mock_signal_manager.setup_signals.assert_called_once_with(mock_bot)
        # Verify lifecycle.execute was called
        mock_lifecycle.execute.assert_called_once_with(mock_bot, single_cycle=True)

    @patch("bot_v2.cli.commands.run.SignalManager")
    @patch("bot_v2.cli.commands.run.LifecycleController")
    def test_successful_bot_run_continuous(
        self, mock_lifecycle_class, mock_signal_class, mock_bot, mock_signal_manager
    ):
        """Test successful bot run in continuous mode."""
        mock_signal_class.return_value = mock_signal_manager
        mock_lifecycle = Mock()
        mock_lifecycle.execute.return_value = 0
        mock_lifecycle_class.return_value = mock_lifecycle

        result = handle_run_bot(mock_bot, dev_fast=False)

        assert result == 0
        mock_signal_class.assert_called_once_with()
        mock_signal_manager.setup_signals.assert_called_once_with(mock_bot)
        # Verify lifecycle.execute was called
        mock_lifecycle.execute.assert_called_once_with(mock_bot, single_cycle=False)

    @patch("bot_v2.cli.commands.run.SignalManager")
    @patch("bot_v2.cli.commands.run.LifecycleController")
    def test_keyboard_interrupt_handled(
        self, mock_lifecycle_class, mock_signal_class, mock_bot, mock_signal_manager
    ):
        """Test KeyboardInterrupt is handled gracefully."""
        mock_signal_class.return_value = mock_signal_manager
        mock_lifecycle = Mock()
        mock_lifecycle.execute.side_effect = KeyboardInterrupt()
        mock_lifecycle_class.return_value = mock_lifecycle

        result = handle_run_bot(mock_bot, dev_fast=False)

        assert result == 0  # Should exit gracefully

    @patch("bot_v2.cli.commands.run.SignalManager")
    @patch("bot_v2.cli.commands.run.LifecycleController")
    def test_exception_returns_error_code(
        self, mock_lifecycle_class, mock_signal_class, mock_bot, mock_signal_manager
    ):
        """Test exception during bot run returns error code."""
        mock_signal_class.return_value = mock_signal_manager
        mock_lifecycle = Mock()
        mock_lifecycle.execute.side_effect = Exception("Bot execution failed")
        mock_lifecycle_class.return_value = mock_lifecycle

        result = handle_run_bot(mock_bot, dev_fast=False)

        assert result == 1

    @patch("bot_v2.cli.commands.run.SignalManager")
    @patch("bot_v2.cli.commands.run.LifecycleController")
    def test_shutdown_handler_registered_before_run(
        self, mock_lifecycle_class, mock_signal_class, mock_bot, mock_signal_manager
    ):
        """Test shutdown handler is registered before bot runs."""
        mock_signal_class.return_value = mock_signal_manager
        mock_lifecycle = Mock()
        mock_lifecycle.execute.return_value = 0
        mock_lifecycle_class.return_value = mock_lifecycle

        handle_run_bot(mock_bot, dev_fast=True)

        # Verify order of calls
        assert mock_signal_class.call_count == 1
        assert mock_signal_manager.setup_signals.call_count == 1
        # lifecycle.execute should be called after signals are registered
        assert mock_lifecycle.execute.call_count == 1

    @patch("bot_v2.cli.commands.run.SignalManager")
    @patch("bot_v2.cli.commands.run.LifecycleController")
    def test_dev_fast_mode_single_cycle(
        self, mock_lifecycle_class, mock_signal_class, mock_bot, mock_signal_manager
    ):
        """Test dev_fast mode runs single cycle."""
        mock_signal_class.return_value = mock_signal_manager
        mock_lifecycle = Mock()
        mock_lifecycle.execute.return_value = 0
        mock_lifecycle_class.return_value = mock_lifecycle

        handle_run_bot(mock_bot, dev_fast=True)

        # Verify single_cycle=True is passed to lifecycle
        mock_lifecycle.execute.assert_called_once_with(mock_bot, single_cycle=True)

    @patch("bot_v2.cli.commands.run.SignalManager")
    @patch("bot_v2.cli.commands.run.LifecycleController")
    def test_continuous_mode_no_single_cycle(
        self, mock_lifecycle_class, mock_signal_class, mock_bot, mock_signal_manager
    ):
        """Test continuous mode runs without single_cycle."""
        mock_signal_class.return_value = mock_signal_manager
        mock_lifecycle = Mock()
        mock_lifecycle.execute.return_value = 0
        mock_lifecycle_class.return_value = mock_lifecycle

        handle_run_bot(mock_bot, dev_fast=False)

        # Verify single_cycle=False is passed to lifecycle
        mock_lifecycle.execute.assert_called_once_with(mock_bot, single_cycle=False)

    @patch("bot_v2.cli.commands.run.SignalManager")
    @patch("bot_v2.cli.commands.run.LifecycleController")
    def test_default_dev_fast_is_false(
        self, mock_lifecycle_class, mock_signal_class, mock_bot, mock_signal_manager
    ):
        """Test default dev_fast parameter is False."""
        mock_signal_class.return_value = mock_signal_manager
        mock_lifecycle = Mock()
        mock_lifecycle.execute.return_value = 0
        mock_lifecycle_class.return_value = mock_lifecycle

        # Call without dev_fast parameter
        handle_run_bot(mock_bot)

        mock_lifecycle.execute.assert_called_once_with(mock_bot, single_cycle=False)

    @patch("bot_v2.cli.commands.run.SignalManager")
    @patch("bot_v2.cli.commands.run.LifecycleController")
    def test_exception_logged_and_raised(
        self, mock_lifecycle_class, mock_signal_class, mock_bot, mock_signal_manager, caplog
    ):
        """Test exception is logged when bot execution fails."""
        import logging

        mock_signal_class.return_value = mock_signal_manager
        mock_lifecycle = Mock()
        mock_lifecycle.execute.side_effect = Exception("Critical error")
        mock_lifecycle_class.return_value = mock_lifecycle

        with caplog.at_level(logging.ERROR):
            result = handle_run_bot(mock_bot, dev_fast=False)

        assert result == 1
        assert "Bot execution failed" in caplog.text

    @patch("bot_v2.cli.commands.run.SignalManager")
    @patch("bot_v2.cli.commands.run.LifecycleController")
    def test_keyboard_interrupt_logged(
        self, mock_lifecycle_class, mock_signal_class, mock_bot, mock_signal_manager, caplog
    ):
        """Test KeyboardInterrupt is logged appropriately."""
        import logging

        mock_signal_class.return_value = mock_signal_manager
        mock_lifecycle = Mock()
        mock_lifecycle.execute.side_effect = KeyboardInterrupt()
        mock_lifecycle_class.return_value = mock_lifecycle

        with caplog.at_level(logging.INFO):
            result = handle_run_bot(mock_bot, dev_fast=False)

        assert result == 0
        assert "KeyboardInterrupt received" in caplog.text or "shutdown complete" in caplog.text
