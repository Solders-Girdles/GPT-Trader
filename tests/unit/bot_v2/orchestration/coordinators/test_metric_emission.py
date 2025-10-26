"""Tests for TelemetryCoordinator metric emission functionality."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot_v2.config.types import Profile
from bot_v2.orchestration.coordinators.telemetry import TelemetryCoordinator


class TestMetricEmission:
    """Test metric emission helpers and payload generation."""

    def test_initialize_creates_telemetry_coordinator_with_dependencies(self, fake_context) -> None:
        """Test initialize method properly creates TelemetryCoordinator with required dependencies."""
        # Initialize with real broker mock
        with patch(
            "bot_v2.orchestration.coordinators.telemetry.CoinbaseBrokerage"
        ) as mock_broker_class:
            mock_broker = MagicMock()
            mock_broker_class.return_value = mock_broker

            result = TelemetryCoordinator.initialize(fake_context)

            # Verify registry updated with new services
            assert "account_manager" in result.registry.extras
            assert "account_telemetry" in result.registry.extras
            assert "intx_portfolio_service" in result.registry.extras
            assert "market_monitor" in result.registry.extras

            # Verify broker was created with context
            mock_broker_class.assert_called_once_with(fake_context)

    def test_initialize_with_none_context_returns_unchanged(self, fake_context) -> None:
        """Test initialize returns unchanged context when context is None."""
        fake_context = None

        result = TelemetryCoordinator.initialize(fake_context)

        assert result is None

    def test_init_market_services_updates_context(self, telemetry_coordinator) -> None:
        """Test init_market_services compatibility helper updates context."""
        original_context = telemetry_coordinator.context

        with patch.object(telemetry_coordinator, "initialize") as mock_init:
            mock_init.return_value = original_context.with_updates(bot_id="updated")

            telemetry_coordinator.init_market_services()

            mock_init.assert_called_once_with(original_context)
            assert telemetry_coordinator.context.bot_id == "updated"

    def test_health_check_returns_healthy_status_with_account_telemetry(
        self, telemetry_coordinator
    ) -> None:
        """Test health_check returns healthy when account telemetry is available."""
        telemetry_coordinator._stream_task = MagicMock()
        telemetry_coordinator._stream_task.done.return_value = False
        telemetry_coordinator._background_tasks = [MagicMock()]

        status = telemetry_coordinator.health_check()

        assert status.healthy is True
        assert status.component == telemetry_coordinator.name
        assert status.details["has_account_telemetry"] is True
        assert status.details["streaming_active"] is True
        assert status.details["background_tasks"] == 1

    def test_health_check_returns_unhealthy_without_account_telemetry(
        self, telemetry_coordinator
    ) -> None:
        """Test health_check returns unhealthy without account telemetry."""
        # Remove account telemetry from registry
        extras = dict(telemetry_coordinator.context.registry.extras)
        extras.pop("account_telemetry", None)
        updated_registry = telemetry_coordinator.context.registry.with_updates(extras=extras)
        telemetry_coordinator.update_context(
            telemetry_coordinator.context.with_updates(registry=updated_registry)
        )

        status = telemetry_coordinator.health_check()

        assert status.healthy is False
        assert status.details["has_account_telemetry"] is False

    async def test_run_account_telemetry_delegates_to_service(self, telemetry_coordinator) -> None:
        """Test run_account_telemetry correctly delegates to account telemetry service."""
        mock_account_telemetry = MagicMock()
        mock_account_telemetry.supports_snapshots.return_value = True
        mock_account_telemetry.run = AsyncMock()

        telemetry_coordinator.context.registry.extras["account_telemetry"] = mock_account_telemetry

        await telemetry_coordinator.run_account_telemetry(interval_seconds=150)

        mock_account_telemetry.supports_snapshots.assert_called_once()
        mock_account_telemetry.run.assert_called_once_with(150)

    async def test_run_account_telemetry_skips_when_no_service(self, telemetry_coordinator) -> None:
        """Test run_account_telemetry skips when no account telemetry service."""
        telemetry_coordinator.context.registry.extras.pop("account_telemetry", None)

        # Should not raise
        await telemetry_coordinator.run_account_telemetry()

    async def test_run_account_telemetry_skips_when_no_snapshot_support(
        self, telemetry_coordinator
    ) -> None:
        """Test run_account_telemetry skips when service doesn't support snapshots."""
        mock_account_telemetry = MagicMock()
        mock_account_telemetry.supports_snapshots.return_value = False

        telemetry_coordinator.context.registry.extras["account_telemetry"] = mock_account_telemetry

        await telemetry_coordinator.run_account_telemetry()

        mock_account_telemetry.supports_snapshots.assert_called_once()
        mock_account_telemetry.run.assert_not_called()

    async def test_start_background_tasks_creates_account_telemetry_task(
        self, telemetry_coordinator
    ) -> None:
        """Test start_background_tasks creates account telemetry task when available."""
        mock_account_telemetry = MagicMock()
        mock_account_telemetry.supports_snapshots.return_value = True

        telemetry_coordinator.context.registry.extras["account_telemetry"] = mock_account_telemetry
        telemetry_coordinator.context.config.account_telemetry_interval = 180

        with patch("asyncio.create_task") as mock_create_task:
            mock_task = MagicMock()
            mock_create_task.return_value = mock_task

            tasks = await telemetry_coordinator.start_background_tasks()

            assert len(tasks) == 1
            assert tasks[0] == mock_task
            mock_create_task.assert_called_once()
            assert len(telemetry_coordinator._background_tasks) == 1

    async def test_start_background_tasks_skips_account_telemetry_when_unavailable(
        self, telemetry_coordinator
    ) -> None:
        """Test start_background_tasks skips account telemetry when unavailable."""
        telemetry_coordinator.context.registry.extras.pop("account_telemetry", None)

        tasks = await telemetry_coordinator.start_background_tasks()

        assert len(tasks) == 0
        assert len(telemetry_coordinator._background_tasks) == 0

    async def test_start_background_tasks_handles_streaming_task_creation_failure(
        self, telemetry_coordinator
    ) -> None:
        """Test start_background_tasks handles streaming task creation failure gracefully."""
        # Mock streaming to be enabled but fail
        telemetry_coordinator.context.config.perps_enable_streaming = True
        telemetry_coordinator.context.config.profile = Profile.PROD
        telemetry_coordinator.context.symbols = ["BTC-PERP"]

        with patch.object(
            telemetry_coordinator, "_start_streaming", side_effect=Exception("Streaming failed")
        ):
            # Should not raise exception
            tasks = await telemetry_coordinator.start_background_tasks()

            # Should still return empty list, not crash
            assert len(tasks) == 0

    async def test_shutdown_stops_streaming_and_calls_super(self, telemetry_coordinator) -> None:
        """Test shutdown stops streaming and calls parent shutdown."""
        with (
            patch.object(
                telemetry_coordinator, "_stop_streaming", new_callable=AsyncMock
            ) as mock_stop,
            patch(
                "bot_v2.orchestration.coordinators.base_coordinator.BaseCoordinator.shutdown",
                new_callable=AsyncMock,
            ) as mock_super_shutdown,
        ):

            await telemetry_coordinator.shutdown()

            mock_stop.assert_called_once()
            mock_super_shutdown.assert_called_once()

    def test_should_enable_streaming_true_in_prod_with_config(self, telemetry_coordinator) -> None:
        """Test _should_enable_streaming returns True in production with streaming enabled."""
        telemetry_coordinator.context.config.perps_enable_streaming = True
        telemetry_coordinator.context.config.profile = Profile.PROD

        result = telemetry_coordinator._should_enable_streaming()

        assert result is True

    def test_should_enable_streaming_true_in_canary_with_config(
        self, telemetry_coordinator
    ) -> None:
        """Test _should_enable_streaming returns True in canary with streaming enabled."""
        telemetry_coordinator.context.config.perps_enable_streaming = True
        telemetry_coordinator.context.config.profile = Profile.CANARY

        result = telemetry_coordinator._should_enable_streaming()

        assert result is True

    def test_should_enable_streaming_false_in_test(self, telemetry_coordinator) -> None:
        """Test _should_enable_streaming returns False in test profile."""
        telemetry_coordinator.context.config.perps_enable_streaming = True
        telemetry_coordinator.context.config.profile = Profile.TEST

        result = telemetry_coordinator._should_enable_streaming()

        assert result is False

    def test_should_enable_streaming_false_when_disabled(self, telemetry_coordinator) -> None:
        """Test _should_enable_streaming returns False when streaming disabled."""
        telemetry_coordinator.context.config.perps_enable_streaming = False
        telemetry_coordinator.context.config.profile = Profile.PROD

        result = telemetry_coordinator._should_enable_streaming()

        assert result is False

    async def test_start_streaming_returns_none_when_no_symbols(
        self, telemetry_coordinator
    ) -> None:
        """Test _start_streaming returns None when no symbols configured."""
        telemetry_coordinator.context.symbols = []

        result = await telemetry_coordinator._start_streaming()

        assert result is None

    async def test_start_streaming_handles_invalid_stream_level(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test _start_streaming handles invalid stream level gracefully."""
        telemetry_coordinator.context.symbols = ["BTC-PERP"]
        telemetry_coordinator.context.config.perps_stream_level = "invalid"

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.create_task.return_value = MagicMock()

            # Set log level to capture warnings
            caplog.set_level("WARNING", logger="bot_v2.orchestration.coordinators.telemetry")

            await telemetry_coordinator._start_streaming()

            # Verify warning logged and level defaulted to 1
            assert "Invalid streaming level; defaulting to 1" in caplog.text
            assert telemetry_coordinator._pending_stream_config[1] == 1

    async def test_start_streaming_returns_none_when_no_running_loop(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test _start_streaming returns None when no running event loop."""
        telemetry_coordinator.context.symbols = ["BTC-PERP"]

        with patch("asyncio.get_running_loop", side_effect=RuntimeError("No running loop")):
            # Set log level to capture debug
            caplog.set_level("DEBUG", logger="bot_v2.orchestration.coordinators.telemetry")

            result = await telemetry_coordinator._start_streaming()

            assert result is None
            assert "No running event loop; streaming will be deferred" in caplog.text

    async def test_stop_streaming_cancels_active_task(self, telemetry_coordinator) -> None:
        """Test _stop_streaming cancels active streaming task."""
        mock_task = MagicMock()
        mock_task.done.return_value = False
        telemetry_coordinator._stream_task = mock_task
        telemetry_coordinator._ws_stop = MagicMock()

        stop_event = telemetry_coordinator._ws_stop

        with patch("asyncio.get_running_loop"):
            await telemetry_coordinator._stop_streaming()

            # Verify stop signal set and task cancelled
            stop_event.set.assert_called_once()
            mock_task.cancel.assert_called_once()
            assert telemetry_coordinator._stream_task is None

    def test_handle_stream_task_completion_handles_cancelled_task(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test _handle_stream_task_completion handles cancelled task correctly."""
        mock_task = MagicMock()
        mock_task.result.side_effect = asyncio.CancelledError()

        # Set log level to capture info
        caplog.set_level("INFO", logger="bot_v2.orchestration.coordinators.telemetry")

        telemetry_coordinator._handle_stream_task_completion(mock_task)

        # Verify task cleaned up and info logged
        assert telemetry_coordinator._stream_task is None
        assert telemetry_coordinator._ws_stop is None
        assert "WS streaming task cancelled" in caplog.text

    def test_handle_stream_task_completion_handles_failed_task(
        self, telemetry_coordinator, caplog
    ) -> None:
        """Test _handle_stream_task_completion handles failed task correctly."""
        mock_task = MagicMock()
        test_error = RuntimeError("Stream connection failed")
        mock_task.result.side_effect = test_error

        # Set log level to capture exceptions
        caplog.set_level("ERROR", logger="bot_v2.orchestration.coordinators.telemetry")

        telemetry_coordinator._handle_stream_task_completion(mock_task)

        # Verify task cleaned up and exception logged
        assert telemetry_coordinator._stream_task is None
        assert telemetry_coordinator._ws_stop is None
        assert "WS streaming task failed" in caplog.text
        assert "Stream connection failed" in caplog.text

    def test_extract_mark_from_message_with_bid_ask(self, telemetry_coordinator) -> None:
        """Test _extract_mark_from_message calculates mark from bid and ask."""
        msg = {"best_bid": "50000", "best_ask": "50100"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        assert result == Decimal("50050")

    def test_extract_mark_from_message_with_last_price(self, telemetry_coordinator) -> None:
        """Test _extract_mark_from_message extracts mark from last price."""
        msg = {"last": "50075"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        assert result == Decimal("50075")

    def test_extract_mark_from_message_with_price_field(self, telemetry_coordinator) -> None:
        """Test _extract_mark_from_message extracts mark from price field."""
        msg = {"price": "50025"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        assert result == Decimal("50025")

    def test_extract_mark_from_message_with_invalid_data(self, telemetry_coordinator) -> None:
        """Test _extract_mark_from_message returns None for invalid data."""
        # Test with negative price
        msg = {"best_bid": "-100", "best_ask": "100"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        assert result is None

    def test_extract_mark_from_message_with_non_numeric_data(self, telemetry_coordinator) -> None:
        """Test _extract_mark_from_message returns None for non-numeric data."""
        msg = {"best_bid": "invalid", "best_ask": "50100"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        assert result is None

    async def test_run_stream_loop_async_handles_cancellation(self, telemetry_coordinator) -> None:
        """Test _run_stream_loop_async handles cancellation correctly."""
        stop_signal = MagicMock()

        with patch.object(telemetry_coordinator, "_run_stream_loop") as mock_run:
            mock_run.side_effect = asyncio.CancelledError()

            with pytest.raises(asyncio.CancelledError):
                await telemetry_coordinator._run_stream_loop_async(["BTC-PERP"], 1, stop_signal)

            # Verify stop signal was set
            stop_signal.set.assert_called_once()

    def test_update_mark_and_metrics_updates_strategy_coordinator(
        self, telemetry_coordinator
    ) -> None:
        """Test _update_mark_and_metrics updates strategy coordinator when available."""
        mock_strategy_coordinator = MagicMock()
        mock_strategy_coordinator.update_mark_window = MagicMock()
        telemetry_coordinator.context.strategy_coordinator = mock_strategy_coordinator

        mark_price = Decimal("50000")

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", mark_price
        )

        mock_strategy_coordinator.update_mark_window.assert_called_once_with("BTC-PERP", mark_price)

    def test_update_mark_and_metrics_fallback_to_runtime_state(self, telemetry_coordinator) -> None:
        """Test _update_mark_and_metrics falls back to runtime state when no strategy coordinator."""
        mark_price = Decimal("50000")

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", mark_price
        )

        # Verify mark window updated in runtime state
        window = telemetry_coordinator.context.runtime_state.mark_windows["BTC-PERP"]
        assert len(window) == 1
        assert window[0] == mark_price

    def test_update_mark_and_metrics_records_market_monitor_update(
        self, telemetry_coordinator
    ) -> None:
        """Test _update_mark_and_metrics records update with market monitor."""
        mock_market_monitor = MagicMock()
        mock_market_monitor.record_update = MagicMock()
        telemetry_coordinator._market_monitor = mock_market_monitor

        mark_price = Decimal("50000")

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", mark_price
        )

        mock_market_monitor.record_update.assert_called_once_with("BTC-PERP")

    def test_update_mark_and_metrics_updates_risk_manager(self, telemetry_coordinator) -> None:
        """Test _update_mark_and_metrics updates risk manager with mark data."""
        mock_risk_manager = MagicMock()
        mock_risk_manager.record_mark_update = MagicMock(return_value=datetime.now(UTC))
        telemetry_coordinator.context.risk_manager = mock_risk_manager

        mark_price = Decimal("50000")

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", mark_price
        )

        mock_risk_manager.record_mark_update.assert_called_once()
        assert "BTC-PERP" in mock_risk_manager.last_mark_update
