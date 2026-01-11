"""Edge-case tests for PositionReconciler."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.monitoring.system.positions import PositionReconciler


def test_normalize_positions_skips_missing_symbol_and_handles_quantity_errors() -> None:
    reconciler = PositionReconciler(event_store=MagicMock(), bot_id="bot-1")
    positions = [
        SimpleNamespace(symbol=None, side="LONG"),
        SimpleNamespace(symbol="BTC-USD", side="LONG"),
    ]

    with (
        patch(
            "gpt_trader.monitoring.system.positions.quantity_from",
            side_effect=RuntimeError("bad quantity"),
        ),
        patch("gpt_trader.monitoring.system.positions.logger") as mock_logger,
    ):
        normalized = reconciler._normalize_positions(positions)

    assert normalized == {}
    mock_logger.exception.assert_called_once()


def test_calculate_diff_tracks_changed_added_removed() -> None:
    reconciler = PositionReconciler(event_store=MagicMock(), bot_id="bot-1")
    previous = {
        "BTC-USD": {"quantity": "1", "side": "LONG"},
        "ETH-USD": {"quantity": "2", "side": "SHORT"},
        "DOGE-USD": {"quantity": "5", "side": "LONG"},
    }
    current = {
        "BTC-USD": {"quantity": "1", "side": "LONG"},
        "ETH-USD": {"quantity": "3", "side": "SHORT"},
        "XRP-USD": {"quantity": "1", "side": "LONG"},
    }

    changes = reconciler._calculate_diff(previous, current)

    assert "BTC-USD" not in changes
    assert changes["ETH-USD"] == {"old": previous["ETH-USD"], "new": current["ETH-USD"]}
    assert changes["XRP-USD"] == {"old": {}, "new": current["XRP-USD"]}
    assert changes["DOGE-USD"] == {"old": previous["DOGE-USD"], "new": {}}


@pytest.mark.asyncio
async def test_fetch_positions_returns_empty_on_error() -> None:
    class _Broker:
        def list_positions(self) -> list[object]:
            raise RuntimeError("boom")

    bot = SimpleNamespace(broker=_Broker())
    reconciler = PositionReconciler(event_store=MagicMock(), bot_id="bot-1")

    assert await reconciler._fetch_positions(bot) == []


def test_emit_position_changes_success_calls_log_and_emit_metric() -> None:
    reconciler = PositionReconciler(event_store=MagicMock(), bot_id="bot-1")
    changes = {
        "BTC-USD": {"old": {}, "new": {"quantity": "1.5", "side": "LONG"}},
        "ETH-USD": {"old": {}, "new": {"quantity": "0", "side": ""}},
    }
    plog = MagicMock()

    with (
        patch("gpt_trader.monitoring.system.get_logger", return_value=plog),
        patch("gpt_trader.monitoring.system.positions.emit_metric") as mock_emit,
    ):
        reconciler._emit_position_changes(None, changes)

    assert plog.log_position_change.call_count == 2
    payload = mock_emit.call_args[0][2]
    assert payload["event_type"] == "position_drift"
    assert payload["changes"] == changes


def test_emit_position_changes_logs_debug_on_failure_and_emits_metric() -> None:
    reconciler = PositionReconciler(event_store=MagicMock(), bot_id="bot-1")
    changes = {"BTC-USD": {"old": {}, "new": {"quantity": "1", "side": "LONG"}}}
    plog = MagicMock()
    plog.log_position_change.side_effect = RuntimeError("boom")

    with (
        patch("gpt_trader.monitoring.system.get_logger", return_value=plog),
        patch("gpt_trader.monitoring.system.positions.emit_metric") as mock_emit,
        patch("gpt_trader.monitoring.system.positions.logger") as mock_logger,
    ):
        reconciler._emit_position_changes(None, changes)

    mock_logger.debug.assert_called_once()
    payload = mock_emit.call_args[0][2]
    assert payload["event_type"] == "position_drift"
