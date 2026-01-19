"""RealTransport edge case coverage tests."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.transports import RealTransport


class TestRealTransportEdgeCases:
    """RealTransport edge case coverage."""

    def test_real_transport_malformed_json_stream(self, mock_bot_config):
        """Test streaming malformed JSON messages."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws

        mock_ws.recv.return_value = '{"invalid": json structure}'

        with pytest.raises(json.JSONDecodeError):
            list(transport.stream())

    def test_real_transport_empty_message_stream(self, mock_bot_config):
        """Test streaming empty messages."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws

        mock_ws.recv.return_value = ""

        with pytest.raises(json.JSONDecodeError):
            list(transport.stream())

    def test_real_transport_none_message_stream(self, mock_bot_config):
        """Test streaming None messages."""
        transport = RealTransport(config=mock_bot_config)
        mock_ws = Mock()
        transport.ws = mock_ws

        mock_ws.recv.return_value = None

        with pytest.raises(TypeError):
            list(transport.stream())
