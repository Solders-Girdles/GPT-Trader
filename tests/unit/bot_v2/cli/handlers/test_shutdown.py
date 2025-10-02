"""Tests for shutdown handler."""

import asyncio
import signal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.cli.handlers.shutdown import ShutdownHandler, ensure_shutdown


@pytest.fixture
def mock_bot():
    """Create a mock PerpsBot."""
    bot = Mock()
    bot.running = True
    bot.shutdown = AsyncMock()
    return bot


@pytest.fixture
def shutdown_handler(mock_bot):
    """Create a ShutdownHandler instance."""
    return ShutdownHandler(mock_bot)


class TestShutdownHandler:
    """Tests for ShutdownHandler class."""

    def test_initialization(self, mock_bot):
        """Test ShutdownHandler initialization."""
        handler = ShutdownHandler(mock_bot)

        assert handler.bot is mock_bot
        assert handler._shutdown_requested is False

    @patch("signal.signal")
    def test_register_signals(self, mock_signal, shutdown_handler):
        """Test signal registration."""
        shutdown_handler.register_signals()

        assert mock_signal.call_count == 2
        mock_signal.assert_any_call(signal.SIGINT, shutdown_handler._signal_handler)
        mock_signal.assert_any_call(signal.SIGTERM, shutdown_handler._signal_handler)

    def test_signal_handler_sets_shutdown_requested(self, shutdown_handler, mock_bot):
        """Test signal handler sets shutdown requested flag."""
        shutdown_handler._signal_handler(signal.SIGINT, None)

        assert shutdown_handler._shutdown_requested is True
        assert mock_bot.running is False

    def test_signal_handler_ignores_duplicate_signals(self, shutdown_handler, mock_bot, caplog):
        """Test signal handler ignores duplicate signals."""
        import logging

        with caplog.at_level(logging.WARNING):
            shutdown_handler._signal_handler(signal.SIGINT, None)
            shutdown_handler._signal_handler(signal.SIGTERM, None)

        assert "already in progress" in caplog.text

    def test_signal_handler_logs_signal_name(self, shutdown_handler, caplog):
        """Test signal handler logs the signal name."""
        import logging

        with caplog.at_level(logging.INFO):
            shutdown_handler._signal_handler(signal.SIGINT, None)

        assert "SIGINT" in caplog.text
        assert "initiating shutdown" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_shutdown_success(self, shutdown_handler, mock_bot, caplog):
        """Test successful shutdown."""
        import logging

        with caplog.at_level(logging.INFO):
            await shutdown_handler.shutdown()

        assert shutdown_handler._shutdown_requested is True
        mock_bot.shutdown.assert_called_once()
        assert "graceful shutdown" in caplog.text.lower()
        assert "completed successfully" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_shutdown_already_initiated(self, shutdown_handler, mock_bot, caplog):
        """Test shutdown when already initiated."""
        import logging

        shutdown_handler._shutdown_requested = True

        with caplog.at_level(logging.DEBUG):
            await shutdown_handler.shutdown()

        assert "already initiated" in caplog.text
        mock_bot.shutdown.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_exception_raised(self, shutdown_handler, mock_bot, caplog):
        """Test exception during shutdown is logged and raised."""
        import logging

        mock_bot.shutdown.side_effect = Exception("Shutdown failed")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception, match="Shutdown failed"):
                await shutdown_handler.shutdown()

        assert "Error during shutdown" in caplog.text

    @patch("asyncio.run")
    def test_shutdown_sync_success(self, mock_asyncio_run, shutdown_handler, caplog):
        """Test synchronous shutdown success."""
        import logging

        with caplog.at_level(logging.INFO):
            shutdown_handler.shutdown_sync()

        mock_asyncio_run.assert_called_once()
        assert "synchronous shutdown" in caplog.text.lower()

    @patch("asyncio.run")
    def test_shutdown_sync_exception(self, mock_asyncio_run, shutdown_handler, caplog):
        """Test exception during synchronous shutdown."""
        import logging

        mock_asyncio_run.side_effect = Exception("Sync shutdown failed")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception, match="Sync shutdown failed"):
                shutdown_handler.shutdown_sync()

        assert "Error during synchronous shutdown" in caplog.text


class TestEnsureShutdown:
    """Tests for ensure_shutdown function."""

    @patch("asyncio.run")
    def test_ensure_shutdown_success(self, mock_asyncio_run, mock_bot, caplog):
        """Test ensure_shutdown runs bot shutdown."""
        import logging

        with caplog.at_level(logging.DEBUG):
            ensure_shutdown(mock_bot)

        # Verify asyncio.run was called with bot.shutdown() coroutine
        assert mock_asyncio_run.call_count == 1
        assert "Ensuring bot shutdown" in caplog.text or "shutdown complete" in caplog.text

    @patch("asyncio.run")
    def test_ensure_shutdown_exception(self, mock_asyncio_run, mock_bot, caplog):
        """Test ensure_shutdown raises exception on failure."""
        import logging

        mock_asyncio_run.side_effect = Exception("Ensure shutdown failed")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception, match="Ensure shutdown failed"):
                ensure_shutdown(mock_bot)

        assert "Error ensuring shutdown" in caplog.text

    @patch("asyncio.run")
    def test_ensure_shutdown_calls_bot_shutdown(self, mock_asyncio_run, mock_bot):
        """Test ensure_shutdown calls bot.shutdown()."""
        ensure_shutdown(mock_bot)

        # Verify asyncio.run was called with bot.shutdown()
        call_args = mock_asyncio_run.call_args[0][0]
        # Should be the coroutine from bot.shutdown()
        assert asyncio.iscoroutine(call_args)


class TestSignalHandling:
    """Tests for signal handling behavior."""

    def test_multiple_signals_only_one_shutdown(self, shutdown_handler, mock_bot):
        """Test multiple signals only trigger one shutdown."""
        shutdown_handler._signal_handler(signal.SIGINT, None)
        shutdown_handler._signal_handler(signal.SIGTERM, None)
        shutdown_handler._signal_handler(signal.SIGINT, None)

        # Only first signal should set running to False
        assert mock_bot.running is False
        assert shutdown_handler._shutdown_requested is True

    def test_signal_handler_stops_bot(self, shutdown_handler, mock_bot):
        """Test signal handler sets bot.running to False."""
        assert mock_bot.running is True

        shutdown_handler._signal_handler(signal.SIGINT, None)

        assert mock_bot.running is False

    def test_signal_names_logged_correctly(self, shutdown_handler, caplog):
        """Test different signal names are logged correctly."""
        import logging

        with caplog.at_level(logging.INFO):
            shutdown_handler._signal_handler(signal.SIGINT, None)
            # Reset for second signal
            shutdown_handler._shutdown_requested = False
            shutdown_handler._signal_handler(signal.SIGTERM, None)

        assert "SIGINT" in caplog.text
        assert "SIGTERM" in caplog.text
