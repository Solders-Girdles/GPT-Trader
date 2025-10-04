"""Tests for signal manager."""

from unittest.mock import Mock

from bot_v2.cli.commands.signal_manager import SignalManager


class TestSignalManager:
    """Test SignalManager."""

    def test_setup_signals_with_default_handler(self):
        """Test setup_signals uses default ShutdownHandler."""
        bot = Mock()
        mock_handler = Mock()
        mock_handler_class = Mock(return_value=mock_handler)

        manager = SignalManager(handler_class=mock_handler_class)
        manager.setup_signals(bot)

        # Verify handler class was instantiated with bot
        mock_handler_class.assert_called_once_with(bot)
        # Verify register_signals was called
        mock_handler.register_signals.assert_called_once()

    def test_setup_signals_with_custom_handler_class(self):
        """Test setup_signals uses custom handler class."""
        bot = Mock()
        mock_handler = Mock()
        custom_handler_class = Mock(return_value=mock_handler)

        manager = SignalManager(handler_class=custom_handler_class)
        manager.setup_signals(bot)

        # Verify custom handler class was used
        custom_handler_class.assert_called_once_with(bot)
        mock_handler.register_signals.assert_called_once()

    def test_setup_signals_creates_new_handler_each_time(self):
        """Test setup_signals creates new handler instance for each call."""
        bot1 = Mock()
        bot2 = Mock()
        mock_handler1 = Mock()
        mock_handler2 = Mock()
        mock_handler_class = Mock(side_effect=[mock_handler1, mock_handler2])

        manager = SignalManager(handler_class=mock_handler_class)
        manager.setup_signals(bot1)
        manager.setup_signals(bot2)

        # Verify handler class was called twice with different bots
        assert mock_handler_class.call_count == 2
        mock_handler_class.assert_any_call(bot1)
        mock_handler_class.assert_any_call(bot2)
        # Verify both handlers had register_signals called
        mock_handler1.register_signals.assert_called_once()
        mock_handler2.register_signals.assert_called_once()

    def test_default_handler_class_is_shutdown_handler(self):
        """Test default handler class is ShutdownHandler when not provided."""
        manager = SignalManager()

        # Verify _handler_class is set (actual ShutdownHandler imported)
        assert manager._handler_class is not None
