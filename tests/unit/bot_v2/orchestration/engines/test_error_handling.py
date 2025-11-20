"""Tests for TelemetryEngine error handling and fallback paths."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot_v2.orchestration.engines.telemetry_coordinator import TelemetryEngine


class TestErrorHandling:
    """Test error handling, fallbacks, and retry logic."""

    async def test_start_streaming_background_early_return_when_disabled(
        self, telemetry_coordinator
    ) -> None:
        """Test start_streaming_background returns early when streaming disabled."""
        telemetry_coordinator.context.config.perps_enable_streaming = False

        telemetry_coordinator.start_streaming_background()

        # Should not create any background tasks
        assert len(telemetry_coordinator._background_tasks) == 0

    async def test_start_streaming_background_schedules_coroutine(
        self, telemetry_coordinator
    ) -> None:
        """Test start_streaming_background schedules streaming coroutine."""
        telemetry_coordinator.context.config.perps_enable_streaming = True
        telemetry_coordinator.context.config.profile = "prod"

        with patch.object(telemetry_coordinator, "_schedule_coroutine") as mock_schedule:
            telemetry_coordinator.start_streaming_background()

            mock_schedule.assert_called_once()

    async def test_stop_streaming_background_schedules_coroutine(
        self, telemetry_coordinator
    ) -> None:
        """Test stop_streaming_background schedules stop coroutine."""
        with patch.object(telemetry_coordinator, "_schedule_coroutine") as mock_schedule:
            telemetry_coordinator.stop_streaming_background()

            mock_schedule.assert_called_once()

    def test_restart_streaming_if_needed_ignores_irrelevant_config_diff(
        self, telemetry_coordinator
    ) -> None:
        """Test restart_streaming_if_needed ignores irrelevant configuration changes."""
        irrelevant_diff = {"some_other_setting": "new_value"}

        with patch.object(telemetry_coordinator, "_schedule_coroutine") as mock_schedule:
            telemetry_coordinator.restart_streaming_if_needed(irrelevant_diff)

            mock_schedule.assert_not_called()

    def test_restart_streaming_if_needed_schedules_restart_for_relevant_diff(
        self, telemetry_coordinator
    ) -> None:
        """Test restart_streaming_if_needed schedules restart for relevant changes."""
        relevant_diff = {"perps_enable_streaming": "true"}

        with patch.object(telemetry_coordinator, "_schedule_coroutine") as mock_schedule:
            telemetry_coordinator.restart_streaming_if_needed(relevant_diff)

            mock_schedule.assert_called_once()

    async def test_restart_streaming_handles_stop_failure_gracefully(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test restart_streaming handles streaming stop failure gracefully."""
        relevant_diff = {"symbols": "BTC-PERP"}

        # Set log level to capture exceptions
        caplog.set_level("ERROR", logger="bot_v2.orchestration.engines.telemetry_coordinator")

        with (
            patch.object(
                telemetry_coordinator, "_stop_streaming", side_effect=Exception("Stop failed")
            ),
            patch.object(
                telemetry_coordinator, "_start_streaming", new_callable=AsyncMock
            ) as mock_start,
        ):

            mock_start.return_value = MagicMock()

            # Execute restart via scheduled coroutine
            with patch.object(telemetry_coordinator, "_schedule_coroutine") as mock_schedule:

                def schedule_side_effect(coro):
                    asyncio.run(coro)

                mock_schedule.side_effect = schedule_side_effect

                telemetry_coordinator.restart_streaming_if_needed(relevant_diff)

            # Verify error logged but restart still attempted
            assert "Failed to stop streaming before restart" in caplog.text
            assert "Stop failed" in caplog.text
            mock_start.assert_called_once()

    async def test_restart_streaming_handles_start_failure_gracefully(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test restart_streaming handles streaming start failure gracefully."""
        relevant_diff = {"perps_enable_streaming": "true"}

        # Set log level to capture exceptions
        caplog.set_level("ERROR", logger="bot_v2.orchestration.engines.telemetry_coordinator")

        with (
            patch.object(telemetry_coordinator, "_stop_streaming", new_callable=AsyncMock),
            patch.object(
                telemetry_coordinator, "_start_streaming", side_effect=Exception("Start failed")
            ),
        ):

            # Execute restart via scheduled coroutine
            with patch.object(telemetry_coordinator, "_schedule_coroutine") as mock_schedule:

                def schedule_side_effect(coro):
                    asyncio.run(coro)

                mock_schedule.side_effect = schedule_side_effect

                telemetry_coordinator.restart_streaming_if_needed(relevant_diff)

            # Verify error logged
            assert "Failed to restart streaming after config change" in caplog.text
            assert "Start failed" in caplog.text

    def test_schedule_coroutine_runs_with_existing_loop(self, telemetry_coordinator) -> None:
        """Test _schedule_coroutine uses existing running loop."""
        coro = MagicMock()

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop

            telemetry_coordinator._schedule_coroutine(coro)

            mock_loop.create_task.assert_called_once_with(coro)

    def test_schedule_coroutine_fallback_to_loop_task(self, telemetry_coordinator) -> None:
        """Test _schedule_coroutine falls back to loop task handle."""
        coro = MagicMock()
        mock_task = MagicMock()
        mock_task_loop = MagicMock()
        mock_task_loop.is_running.return_value = True

        telemetry_coordinator._loop_task_handle = mock_task

        with (
            patch("asyncio.get_running_loop", side_effect=RuntimeError("No loop")),
            patch.object(mock_task, "get_loop", return_value=mock_task_loop),
        ):

            telemetry_coordinator._schedule_coroutine(coro)

            mock_task_loop.call_soon_threadsafe.assert_called_once()

    def test_schedule_coroutine_runs_coroutine_when_no_loop(self, telemetry_coordinator) -> None:
        """Test _schedule_coroutine runs coroutine directly when no loop available."""
        coro = AsyncMock()

        with (
            patch("asyncio.get_running_loop", side_effect=RuntimeError("No loop")),
            patch("asyncio.run") as mock_run,
        ):

            telemetry_coordinator._schedule_coroutine(coro)

            mock_run.assert_called_once_with(coro)

    def test_schedule_coroutine_handles_loop_task_get_loop_exception(
        self, telemetry_coordinator
    ) -> None:
        """Test _schedule_coroutine handles exception when getting loop from task."""
        coro = MagicMock()
        mock_task = MagicMock()

        telemetry_coordinator._loop_task_handle = mock_task

        with (
            patch("asyncio.get_running_loop", side_effect=RuntimeError("No loop")),
            patch.object(mock_task, "get_loop", side_effect=Exception("Task error")),
            patch("asyncio.run") as mock_run,
        ):

            telemetry_coordinator._schedule_coroutine(coro)

            mock_run.assert_called_once_with(coro)

    async def test_run_stream_loop_handles_no_broker_error(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test _run_stream_loop handles missing broker gracefully."""
        telemetry_coordinator.context.broker = None
        stop_signal = MagicMock()

        # Set log level to capture errors
        caplog.set_level("ERROR", logger="bot_v2.orchestration.engines.telemetry_coordinator")

        telemetry_coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

        assert "Cannot start streaming: no broker available" in caplog.text

    def test_run_stream_loop_handles_orderbook_stream_failure_with_fallback(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test _run_stream_loop falls back to trades when orderbook fails."""
        mock_broker = MagicMock()
        mock_broker.stream_orderbook.side_effect = Exception("Orderbook unavailable")
        mock_broker.stream_trades.return_value = iter([{"symbol": "BTC-PERP", "price": "50000"}])
        telemetry_coordinator.context.broker = mock_broker
        stop_signal = MagicMock()

        # Set log level to capture warnings
        caplog.set_level("WARNING", logger="bot_v2.orchestration.engines.telemetry_coordinator")

        telemetry_coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

        # Verify fallback attempted
        assert "Orderbook stream unavailable, falling back to trades" in caplog.text
        assert "Orderbook unavailable" in caplog.text
        mock_broker.stream_trades.assert_called_once_with(["BTC-PERP"])

    def test_run_stream_loop_handles_trade_stream_failure(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test _run_stream_loop handles trade stream failure."""
        mock_broker = MagicMock()
        mock_broker.stream_orderbook.side_effect = Exception("Orderbook unavailable")
        mock_broker.stream_trades.side_effect = Exception("Trades unavailable")
        telemetry_coordinator.context.broker = mock_broker
        stop_signal = MagicMock()

        # Set log level to capture errors
        caplog.set_level("ERROR", logger="bot_v2.orchestration.engines.telemetry_coordinator")

        telemetry_coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

        # Verify both streams failed
        assert "Failed to start streaming trades" in caplog.text
        assert "Trades unavailable" in caplog.text

    def test_run_stream_loop_handles_message_processing_error_with_metric(
        self, telemetry_coordinator
    ) -> None:
        """Test _run_stream_loop emits error metric when message processing fails."""
        mock_broker = MagicMock()

        # Create a generator that raises during iteration
        def faulty_stream():
            yield {"symbol": "BTC-PERP", "price": "50000"}
            raise RuntimeError("Processing error")

        mock_broker.stream_orderbook.return_value = faulty_stream()
        telemetry_coordinator.context.broker = mock_broker
        stop_signal = MagicMock()

        with patch("bot_v2.orchestration.engines.telemetry_coordinator.emit_metric") as mock_emit:
            telemetry_coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

            # Verify error metric emitted
            mock_emit.assert_called()
            error_calls = [
                call for call in mock_emit.call_args_list if "ws_stream_error" in str(call)
            ]
            assert len(error_calls) > 0

    def test_run_stream_loop_handles_stop_signal_during_iteration(
        self, telemetry_coordinator
    ) -> None:
        """Test _run_stream_loop respects stop signal during message iteration."""
        mock_broker = MagicMock()

        # Create a stream that continues indefinitely
        def infinite_stream():
            while True:
                yield {"symbol": "BTC-PERP", "price": "50000"}

        mock_broker.stream_orderbook.return_value = infinite_stream()
        telemetry_coordinator.context.broker = mock_broker

        # Set stop signal immediately
        stop_signal = MagicMock()
        stop_signal.is_set.return_value = True

        telemetry_coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

        # Should exit early due to stop signal
        # We can't easily verify this without more complex mocking, but the test ensures no infinite loop

    def test_run_stream_loop_filters_invalid_messages(self, telemetry_coordinator) -> None:
        """Test _run_stream_loop filters out invalid message formats."""
        mock_broker = MagicMock()
        messages = [
            None,  # Invalid: not a dict
            {},  # Invalid: missing symbol
            {"symbol": ""},  # Invalid: empty symbol
            {"product_id": None},  # Invalid: None symbol
            {"symbol": "BTC-PERP", "price": "invalid"},  # Invalid: can't convert to Decimal
            {"symbol": "BTC-PERP", "price": "-100"},  # Invalid: negative price
            {
                "symbol": "BTC-PERP",
                "best_bid": "50000",
            },  # Invalid: missing ask for mark calculation
        ]
        mock_broker.stream_orderbook.return_value = iter(messages)
        telemetry_coordinator.context.broker = mock_broker
        stop_signal = MagicMock()

        # Should not raise exception and should filter all invalid messages
        telemetry_coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

    def test_run_stream_loop_emits_exit_metric_on_completion(self, telemetry_coordinator) -> None:
        """Test _run_stream_loop emits exit metric when stream completes."""
        mock_broker = MagicMock()
        mock_broker.stream_orderbook.return_value = iter([{"symbol": "BTC-PERP", "price": "50000"}])
        telemetry_coordinator.context.broker = mock_broker
        stop_signal = MagicMock()

        with patch("bot_v2.orchestration.engines.telemetry_coordinator.emit_metric") as mock_emit:
            telemetry_coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

            # Verify exit metric emitted
            exit_calls = [
                call for call in mock_emit.call_args_list if "ws_stream_exit" in str(call)
            ]
            assert len(exit_calls) > 0

    def test_update_mark_and_metrics_handles_strategy_coordinator_error(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test _update_mark_and_metrics handles strategy coordinator update error."""
        mock_strategy_coordinator = MagicMock()
        mock_strategy_coordinator.update_mark_window.side_effect = Exception("Strategy error")
        telemetry_coordinator.context.strategy_coordinator = mock_strategy_coordinator

        # Set log level to capture debug
        caplog.set_level("DEBUG", logger="bot_v2.orchestration.engines.telemetry_coordinator")

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", Decimal("50000")
        )

        # Verify error logged but processing continues
        assert "Failed to update mark window" in caplog.text
        assert "Strategy error" in caplog.text

    def test_update_mark_and_metrics_handles_market_monitor_error(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test _update_mark_and_metrics handles market monitor recording error."""
        mock_market_monitor = MagicMock()
        mock_market_monitor.record_update.side_effect = Exception("Monitor error")
        telemetry_coordinator._market_monitor = mock_market_monitor

        # Set log level to capture debug
        caplog.set_level("DEBUG", logger="bot_v2.orchestration.engines.telemetry_coordinator")

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", Decimal("50000")
        )

        # Verify error logged but processing continues
        assert "Failed to record market update" in caplog.text
        assert "Monitor error" in caplog.text

    def test_update_mark_and_metrics_handles_risk_manager_error(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test _update_mark_and_metrics handles risk manager update error."""
        mock_risk_manager = MagicMock()
        mock_risk_manager.record_mark_update.side_effect = Exception("Risk error")
        telemetry_coordinator.context.risk_manager = mock_risk_manager

        # Set log level to capture exceptions
        caplog.set_level("ERROR", logger="bot_v2.orchestration.engines.telemetry_coordinator")

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", Decimal("50000")
        )

        # Verify error logged
        assert "WS mark update bookkeeping failed" in caplog.text
        assert "Risk error" in caplog.text

    def test_update_mark_and_metrics_handles_missing_risk_manager_recording_method(
        self, telemetry_coordinator
    ) -> None:
        """Test _update_mark_and_metrics handles missing record_mark_update method."""
        mock_risk_manager = MagicMock()
        # Don't set record_mark_update method (it doesn't exist by default)
        del mock_risk_manager.record_mark_update
        telemetry_coordinator.context.risk_manager = mock_risk_manager

        # Should not raise exception
        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", Decimal("50000")
        )

        # Verify timestamp stored directly
        assert "BTC-PERP" in mock_risk_manager.last_mark_update

    async def test_initialize_handles_broker_creation_failure(self, fake_context) -> None:
        """Test initialize handles broker creation failure gracefully."""
        with patch(
            "bot_v2.orchestration.engines.telemetry_coordinator.CoinbaseBrokerage",
            side_effect=Exception("Broker creation failed"),
        ):
            # Should propagate exception - broker creation is critical
            with pytest.raises(Exception, match="Broker creation failed"):
                TelemetryEngine.initialize(fake_context)

    def test_initialize_handles_market_monitor_creation_failure(self, fake_context) -> None:
        """Test initialize handles market monitor creation failure gracefully."""
        with (
            patch(
                "bot_v2.orchestration.engines.telemetry_coordinator.CoinbaseBrokerage"
            ) as mock_broker_class,
            patch(
                "bot_v2.orchestration.engines.telemetry_coordinator.MarketActivityMonitor",
                side_effect=Exception("Monitor creation failed"),
            ),
        ):

            mock_broker = MagicMock()
            mock_broker_class.return_value = mock_broker

            # Should still create coordinator even if market monitor fails
            result = TelemetryEngine.initialize(fake_context)

            # Should have other services but market monitor might be None
            assert "account_manager" in result.registry.extras
            assert "account_telemetry" in result.registry.extras
