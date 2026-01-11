"""Edge coverage for CoinbaseWebSocket message handling."""

from __future__ import annotations

import json
from unittest.mock import patch

from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket


def test_subscribe_user_events_without_credentials_logs_warning() -> None:
    ws = CoinbaseWebSocket(api_key=None, private_key=None)

    with patch("gpt_trader.features.brokerages.coinbase.ws.logger") as mock_logger:
        ws.subscribe_user_events()

    mock_logger.warning.assert_called_once_with(
        "Cannot subscribe to user events without API credentials"
    )
    assert ws.subscriptions == []


def test_on_message_heartbeat_updates_timestamps() -> None:
    ws = CoinbaseWebSocket()

    with patch(
        "gpt_trader.features.brokerages.coinbase.ws.time.time",
        side_effect=[1.0, 2.0],
    ):
        ws._on_message(None, json.dumps({"channel": "heartbeats"}))

    assert ws._last_message_ts == 1.0
    assert ws._last_heartbeat_ts == 2.0


def test_on_message_sequence_gap_increments_and_passes_flag() -> None:
    captured: list[dict] = []

    def _capture(message: dict) -> None:
        captured.append(message)

    ws = CoinbaseWebSocket(on_message=_capture)

    with patch("gpt_trader.features.brokerages.coinbase.ws.time.time", return_value=1.0):
        ws._on_message(None, json.dumps({"sequence": 1, "channel": "ticker"}))
        ws._on_message(None, json.dumps({"sequence": 3, "channel": "ticker"}))

    assert ws._gap_count == 1
    assert captured[-1].get("gap_detected") is True


def test_invalid_json_logs_error_and_preserves_last_message_ts() -> None:
    ws = CoinbaseWebSocket()
    ws._last_message_ts = 10.0

    with patch("gpt_trader.features.brokerages.coinbase.ws.logger") as mock_logger:
        ws._on_message(None, "not-json")

    assert ws._last_message_ts == 10.0
    mock_logger.error.assert_called_once()
