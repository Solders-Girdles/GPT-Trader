"""Focused transport environment option parsing tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import websocket

from gpt_trader.features.brokerages.coinbase.transports import RealTransport


class TestTransportEnvironmentOptions:
    """Test environment-driven options in RealTransport.connect."""

    def test_transport_environment_variable_parsing(self, mock_bot_config, monkeypatch):
        """Test various environment variable parsing scenarios."""
        for truthy_value in ["1", "true", "yes", "on", "TRUE", "Yes", "ON"]:
            monkeypatch.setenv("COINBASE_WS_ENABLE_TRACE", truthy_value)

            transport = RealTransport(config=mock_bot_config)

            mock_create = Mock()
            monkeypatch.setattr(websocket, "create_connection", mock_create)
            mock_trace = Mock()
            monkeypatch.setattr(websocket, "enableTrace", mock_trace)
            mock_ws = Mock()
            mock_create.return_value = mock_ws
            transport.connect("wss://test.example.com")

            mock_trace.assert_called_once_with(True)

    def test_transport_empty_subprotocols(
        self,
        mock_bot_config,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test empty subprotocols configuration."""
        monkeypatch.setenv("COINBASE_WS_SUBPROTOCOLS", "   ,  , ")
        monkeypatch.setattr(websocket, "enableTrace", Mock())

        transport = RealTransport(config=mock_bot_config)

        mock_create = Mock()
        monkeypatch.setattr(websocket, "create_connection", mock_create)
        mock_ws = Mock()
        mock_create.return_value = mock_ws

        transport.connect("wss://test.example.com")

        call_args = mock_create.call_args[1]
        assert "subprotocols" not in call_args

    def test_transport_mixed_whitespace_subprotocols(
        self,
        mock_bot_config,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test subprotocols with mixed whitespace."""
        monkeypatch.setenv("COINBASE_WS_SUBPROTOCOLS", " v1 , v2 , v3 ")
        monkeypatch.setattr(websocket, "enableTrace", Mock())

        transport = RealTransport(config=mock_bot_config)

        mock_create = Mock()
        monkeypatch.setattr(websocket, "create_connection", mock_create)
        mock_ws = Mock()
        mock_create.return_value = mock_ws

        transport.connect("wss://test.example.com")

        expected_subprotocols = ["v1", "v2", "v3"]
        call_args = mock_create.call_args[1]
        assert call_args["subprotocols"] == expected_subprotocols
