from unittest.mock import MagicMock, patch

from gpt_trader.tui.app import TraderApp


class TestResourceCleanup:
    """Tests for proper resource cleanup on TUI shutdown."""

    def test_coinbase_client_has_close_method(self):
        from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase

        client = CoinbaseClientBase()
        assert hasattr(client, "close")
        assert hasattr(client, "__enter__")
        assert hasattr(client, "__exit__")

        client.close()

    def test_coinbase_client_context_manager(self):
        from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase

        with CoinbaseClientBase() as client:
            assert client.session is not None

    def test_websocket_has_close_method(self):
        from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket

        ws = CoinbaseWebSocket()
        assert hasattr(ws, "close")
        assert hasattr(ws, "__enter__")
        assert hasattr(ws, "__exit__")
        assert hasattr(ws, "_shutdown")

        ws.close()

    def test_websocket_close_is_idempotent(self):
        from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket

        ws = CoinbaseWebSocket()
        ws.close()
        ws.close()
        assert ws._closed is True
        assert ws._shutdown.is_set() is True
        assert ws.ws is None
        assert ws.wst is None

    def test_performance_service_cleanup(self):
        from gpt_trader.tui.services.performance_service import (
            TuiPerformanceService,
            clear_tui_performance_service,
            get_tui_performance_service,
            set_tui_performance_service,
        )

        service = TuiPerformanceService(enabled=False)
        set_tui_performance_service(service)
        assert get_tui_performance_service() is service

        clear_tui_performance_service()

        new_service = get_tui_performance_service()
        assert new_service is not service

        clear_tui_performance_service()

    def test_app_cleanup_bot_resources_closes_client(self, mock_bot):
        mock_client = MagicMock()
        mock_bot.client = mock_client

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()

        mock_client.close.assert_called_once()

    def test_app_cleanup_bot_resources_closes_websocket(self, mock_bot):
        mock_ws = MagicMock()
        mock_bot.websocket = mock_ws

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()

        mock_ws.close.assert_called_once()

    def test_app_cleanup_bot_resources_closes_engine_client(self, mock_bot):
        mock_engine_client = MagicMock()
        mock_bot.engine.client = mock_engine_client

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()

        mock_engine_client.close.assert_called_once()

    def test_app_cleanup_handles_missing_resources(self, mock_bot):
        if hasattr(mock_bot, "client"):
            del mock_bot.client
        if hasattr(mock_bot, "websocket"):
            del mock_bot.websocket

        app = TraderApp(mock_bot)
        app._cleanup_bot_resources()
        assert not hasattr(mock_bot, "client")
        assert not hasattr(mock_bot, "websocket")

    def test_app_cleanup_handles_close_errors(self, mock_bot):
        mock_client = MagicMock()
        mock_client.close.side_effect = RuntimeError("Close failed")
        mock_bot.client = mock_client

        app = TraderApp(mock_bot)
        with patch("gpt_trader.tui.app_lifecycle.logger") as mock_logger:
            app._cleanup_bot_resources()
            mock_logger.warning.assert_called_once()

    def test_websocket_reconnect_limit_configurable(self):
        from gpt_trader.config.constants import MAX_WS_RECONNECT_ATTEMPTS

        assert MAX_WS_RECONNECT_ATTEMPTS == 10
