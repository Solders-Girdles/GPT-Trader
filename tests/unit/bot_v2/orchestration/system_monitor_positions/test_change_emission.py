"""Tests for position change emission and integration with external services."""

from __future__ import annotations

from bot_v2.orchestration.system_monitor_positions import PositionReconciler


class TestChangeEmission:
    """Test position change emission and external service integration."""

    def test_emit_position_changes_happy_path(
        self, reconciler: PositionReconciler, fake_bot, fake_plog, emit_metric_spy, caplog
    ) -> None:
        """Test _emit_position_changes successfully processes changes."""
        changes = {
            "BTC-PERP": {
                "old": {"quantity": "0.3", "side": "long"},
                "new": {"quantity": "0.5", "side": "long"},
            },
            "ETH-PERP": {
                "old": {},
                "new": {"quantity": "1.0", "side": "short"},
            },
        }

        # Set log level to capture info messages
        caplog.set_level("INFO", logger="bot_v2.orchestration.system_monitor_positions")

        reconciler._emit_position_changes(fake_bot, changes)

        # Verify info log emitted
        assert "Position changes detected" in caplog.text
        assert "change_count=2" in caplog.text

        # Verify plog called for each change
        assert fake_plog.log_position_change.call_count == 2

        # Check BTC-PERP call
        btc_call = fake_plog.log_position_change.call_args_list[0]
        assert btc_call.kwargs["symbol"] == "BTC-PERP"
        assert btc_call.kwargs["side"] == "long"
        assert btc_call.kwargs["size"] == 0.5

        # Check ETH-PERP call
        eth_call = fake_plog.log_position_change.call_args_list[1]
        assert eth_call.kwargs["symbol"] == "ETH-PERP"
        assert eth_call.kwargs["side"] == "short"
        assert eth_call.kwargs["size"] == 1.0

        # Verify emit_metric called with correct payload
        emit_metric_spy.assert_called_once()
        metric_call = emit_metric_spy.call_args
        assert metric_call.args[0] == reconciler._event_store
        assert metric_call.args[1] == reconciler._bot_id
        metric_data = metric_call.args[2]
        assert metric_data["event_type"] == "position_drift"
        assert metric_data["changes"] == changes

    def test_emit_position_changes_closed_position(
        self, reconciler: PositionReconciler, fake_bot, fake_plog, emit_metric_spy
    ) -> None:
        """Test _emit_position_changes handles closed positions correctly."""
        changes = {
            "BTC-PERP": {
                "old": {"quantity": "0.5", "side": "long"},
                "new": {},  # Position closed
            }
        }

        reconciler._emit_position_changes(fake_bot, changes)

        # Verify plog called with size 0 and empty side
        fake_plog.log_position_change.assert_called_once()
        call = fake_plog.log_position_change.call_args
        assert call.kwargs["symbol"] == "BTC-PERP"
        assert call.kwargs["side"] == ""
        assert call.kwargs["size"] == 0.0

        # Verify metric still emitted
        emit_metric_spy.assert_called_once()

    def test_emit_position_changes_plog_failure(
        self, reconciler: PositionReconciler, fake_bot, fake_plog, emit_metric_spy, caplog
    ) -> None:
        """Test _emit_position_changes handles plog failures gracefully."""
        changes = {
            "BTC-PERP": {
                "old": {"quantity": "0.3", "side": "long"},
                "new": {"quantity": "0.5", "side": "long"},
            }
        }

        # Set log levels to capture both info and debug messages
        caplog.set_level("INFO", logger="bot_v2.orchestration.system_monitor_positions")
        caplog.set_level("DEBUG", logger="bot_v2.orchestration.system_monitor_positions")

        # Make plog raise an exception
        fake_plog.log_position_change.side_effect = RuntimeError("PLog error")

        reconciler._emit_position_changes(fake_bot, changes)

        # Should still have info log
        assert "Position changes detected" in caplog.text

        # Should have debug log about plog failure
        assert "Failed to log position change metric" in caplog.text
        assert "PLog error" in caplog.text

        # Verify emit_metric still called despite plog failure
        emit_metric_spy.assert_called_once()
        metric_call = emit_metric_spy.call_args
        metric_data = metric_call.args[2]
        assert metric_data["event_type"] == "position_drift"
        assert metric_data["changes"] == changes

    def test_emit_position_changes_empty_changes(
        self, reconciler: PositionReconciler, fake_bot, fake_plog, emit_metric_spy, caplog
    ) -> None:
        """Test _emit_position_changes handles empty changes gracefully."""
        changes = {}

        # Set log level to capture info messages
        caplog.set_level("INFO", logger="bot_v2.orchestration.system_monitor_positions")

        reconciler._emit_position_changes(fake_bot, changes)

        # Should still log the info message with 0 changes
        assert "Position changes detected" in caplog.text
        assert "change_count=0" in caplog.text

        # Should not call plog for empty changes
        fake_plog.log_position_change.assert_not_called()

        # Should still emit metric with empty changes
        emit_metric_spy.assert_called_once()
        metric_call = emit_metric_spy.call_args
        metric_data = metric_call.args[2]
        assert metric_data["event_type"] == "position_drift"
        assert metric_data["changes"] == {}

    def test_emit_position_changes_mixed_new_and_closed(
        self, reconciler: PositionReconciler, fake_bot, fake_plog, emit_metric_spy
    ) -> None:
        """Test _emit_position_changes handles mix of new and closed positions."""
        changes = {
            "BTC-PERP": {
                "old": {"quantity": "0.5", "side": "long"},
                "new": {},  # Closed
            },
            "ETH-PERP": {
                "old": {},
                "new": {"quantity": "1.0", "side": "short"},  # New
            },
            "SOL-PERP": {
                "old": {"quantity": "100", "side": "long"},
                "new": {"quantity": "120", "side": "long"},  # Modified
            },
        }

        reconciler._emit_position_changes(fake_bot, changes)

        # Should have 3 plog calls
        assert fake_plog.log_position_change.call_count == 3

        calls = fake_plog.log_position_change.call_args_list

        # BTC-PERP (closed)
        assert calls[0].kwargs["symbol"] == "BTC-PERP"
        assert calls[0].kwargs["size"] == 0.0
        assert calls[0].kwargs["side"] == ""

        # ETH-PERP (new)
        assert calls[1].kwargs["symbol"] == "ETH-PERP"
        assert calls[1].kwargs["size"] == 1.0
        assert calls[1].kwargs["side"] == "short"

        # SOL-PERP (modified)
        assert calls[2].kwargs["symbol"] == "SOL-PERP"
        assert calls[2].kwargs["size"] == 120.0
        assert calls[2].kwargs["side"] == "long"

        # Verify metric emitted with all changes
        emit_metric_spy.assert_called_once()
        metric_call = emit_metric_spy.call_args
        metric_data = metric_call.args[2]
        assert len(metric_data["changes"]) == 3
        assert "BTC-PERP" in metric_data["changes"]
        assert "ETH-PERP" in metric_data["changes"]
        assert "SOL-PERP" in metric_data["changes"]

    def test_emit_position_changes_invalid_quantity_values(
        self, reconciler: PositionReconciler, fake_bot, fake_plog, emit_metric_spy
    ) -> None:
        """Test _emit_position_changes handles invalid quantity values gracefully."""
        changes = {
            "BTC-PERP": {
                "old": {"quantity": "0.3", "side": "long"},
                "new": {"quantity": None, "side": "long"},  # None quantity
            },
            "ETH-PERP": {
                "old": {"quantity": "", "side": "short"},
                "new": {"quantity": "1.0", "side": "short"},  # Empty old quantity
            },
        }

        reconciler._emit_position_changes(fake_bot, changes)

        # Should have 2 calls despite invalid values
        assert fake_plog.log_position_change.call_count == 2

        calls = fake_plog.log_position_change.call_args_list

        # BTC-PERP (None quantity -> 0.0)
        assert calls[0].kwargs["symbol"] == "BTC-PERP"
        assert calls[0].kwargs["size"] == 0.0

        # ETH-PERP (valid quantity)
        assert calls[1].kwargs["symbol"] == "ETH-PERP"
        assert calls[1].kwargs["size"] == 1.0

        # Metric still emitted
        emit_metric_spy.assert_called_once()
