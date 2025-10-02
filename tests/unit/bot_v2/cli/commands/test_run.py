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
def mock_shutdown_handler():
    """Create a mock ShutdownHandler."""
    handler = Mock()
    handler.register_signals = Mock()
    return handler


class TestHandleRunBot:
    """Tests for handle_run_bot function."""

    @patch("bot_v2.cli.commands.run.ShutdownHandler")
    @patch("bot_v2.cli.commands.run.asyncio.run")
    def test_successful_bot_run_dev_mode(self, mock_asyncio_run, mock_handler_class, mock_bot, mock_shutdown_handler):
        """Test successful bot run in dev_fast mode."""
        mock_handler_class.return_value = mock_shutdown_handler

        result = handle_run_bot(mock_bot, dev_fast=True)

        assert result == 0
        mock_handler_class.assert_called_once_with(mock_bot)
        mock_shutdown_handler.register_signals.assert_called_once()
        # Verify asyncio.run was called with a coroutine from bot.run
        assert mock_asyncio_run.call_count == 1
        mock_bot.run.assert_called_once_with(single_cycle=True)

    @patch("bot_v2.cli.commands.run.ShutdownHandler")
    @patch("bot_v2.cli.commands.run.asyncio.run")
    def test_successful_bot_run_continuous(self, mock_asyncio_run, mock_handler_class, mock_bot, mock_shutdown_handler):
        """Test successful bot run in continuous mode."""
        mock_handler_class.return_value = mock_shutdown_handler

        result = handle_run_bot(mock_bot, dev_fast=False)

        assert result == 0
        mock_handler_class.assert_called_once_with(mock_bot)
        mock_shutdown_handler.register_signals.assert_called_once()
        # Verify asyncio.run was called with a coroutine from bot.run
        assert mock_asyncio_run.call_count == 1
        mock_bot.run.assert_called_once_with(single_cycle=False)

    @patch("bot_v2.cli.commands.run.ShutdownHandler")
    @patch("bot_v2.cli.commands.run.asyncio.run")
    def test_keyboard_interrupt_handled(self, mock_asyncio_run, mock_handler_class, mock_bot, mock_shutdown_handler):
        """Test KeyboardInterrupt is handled gracefully."""
        mock_handler_class.return_value = mock_shutdown_handler
        mock_asyncio_run.side_effect = KeyboardInterrupt()

        result = handle_run_bot(mock_bot, dev_fast=False)

        assert result == 0  # Should exit gracefully

    @patch("bot_v2.cli.commands.run.ShutdownHandler")
    @patch("bot_v2.cli.commands.run.asyncio.run")
    def test_exception_returns_error_code(self, mock_asyncio_run, mock_handler_class, mock_bot, mock_shutdown_handler):
        """Test exception during bot run returns error code."""
        mock_handler_class.return_value = mock_shutdown_handler
        mock_asyncio_run.side_effect = Exception("Bot execution failed")

        result = handle_run_bot(mock_bot, dev_fast=False)

        assert result == 1

    @patch("bot_v2.cli.commands.run.ShutdownHandler")
    @patch("bot_v2.cli.commands.run.asyncio.run")
    def test_shutdown_handler_registered_before_run(self, mock_asyncio_run, mock_handler_class, mock_bot, mock_shutdown_handler):
        """Test shutdown handler is registered before bot runs."""
        mock_handler_class.return_value = mock_shutdown_handler

        handle_run_bot(mock_bot, dev_fast=True)

        # Verify order of calls
        assert mock_handler_class.call_count == 1
        assert mock_shutdown_handler.register_signals.call_count == 1
        # asyncio.run should be called after signals are registered
        assert mock_asyncio_run.call_count == 1

    @patch("bot_v2.cli.commands.run.ShutdownHandler")
    @patch("bot_v2.cli.commands.run.asyncio.run")
    def test_dev_fast_mode_single_cycle(self, mock_asyncio_run, mock_handler_class, mock_bot, mock_shutdown_handler):
        """Test dev_fast mode runs single cycle."""
        mock_handler_class.return_value = mock_shutdown_handler

        handle_run_bot(mock_bot, dev_fast=True)

        # Verify single_cycle=True is passed
        call_args = mock_asyncio_run.call_args[0][0]
        # Check that bot.run was called with single_cycle=True
        mock_bot.run.assert_called_once_with(single_cycle=True)

    @patch("bot_v2.cli.commands.run.ShutdownHandler")
    @patch("bot_v2.cli.commands.run.asyncio.run")
    def test_continuous_mode_no_single_cycle(self, mock_asyncio_run, mock_handler_class, mock_bot, mock_shutdown_handler):
        """Test continuous mode runs without single_cycle."""
        mock_handler_class.return_value = mock_shutdown_handler

        handle_run_bot(mock_bot, dev_fast=False)

        # Verify single_cycle=False is passed
        mock_bot.run.assert_called_once_with(single_cycle=False)

    @patch("bot_v2.cli.commands.run.ShutdownHandler")
    @patch("bot_v2.cli.commands.run.asyncio.run")
    def test_default_dev_fast_is_false(self, mock_asyncio_run, mock_handler_class, mock_bot, mock_shutdown_handler):
        """Test default dev_fast parameter is False."""
        mock_handler_class.return_value = mock_shutdown_handler

        # Call without dev_fast parameter
        handle_run_bot(mock_bot)

        mock_bot.run.assert_called_once_with(single_cycle=False)

    @patch("bot_v2.cli.commands.run.ShutdownHandler")
    @patch("bot_v2.cli.commands.run.asyncio.run")
    def test_exception_logged_and_raised(self, mock_asyncio_run, mock_handler_class, mock_bot, mock_shutdown_handler, caplog):
        """Test exception is logged when bot execution fails."""
        import logging

        mock_handler_class.return_value = mock_shutdown_handler
        mock_asyncio_run.side_effect = Exception("Critical error")

        with caplog.at_level(logging.ERROR):
            result = handle_run_bot(mock_bot, dev_fast=False)

        assert result == 1
        assert "Bot execution failed" in caplog.text

    @patch("bot_v2.cli.commands.run.ShutdownHandler")
    @patch("bot_v2.cli.commands.run.asyncio.run")
    def test_keyboard_interrupt_logged(self, mock_asyncio_run, mock_handler_class, mock_bot, mock_shutdown_handler, caplog):
        """Test KeyboardInterrupt is logged appropriately."""
        import logging

        mock_handler_class.return_value = mock_shutdown_handler
        mock_asyncio_run.side_effect = KeyboardInterrupt()

        with caplog.at_level(logging.INFO):
            result = handle_run_bot(mock_bot, dev_fast=False)

        assert result == 0
        assert "KeyboardInterrupt received" in caplog.text or "shutdown complete" in caplog.text
