"""Simple tests for TelemetryCoordinator core functionality."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from bot_v2.config.types import Profile
from bot_v2.orchestration.coordinators.telemetry import TelemetryCoordinator


class TestTelemetryCoordinatorCore:
    """Test core TelemetryCoordinator functionality with minimal dependencies."""

    def test_telemetry_coordinator_initialization(self, fake_context) -> None:
        """Test TelemetryCoordinator can be initialized with basic context."""
        coordinator = TelemetryCoordinator(context=fake_context)

        assert coordinator.name == "telemetry"
        assert coordinator.context == fake_context

    def test_health_check_without_account_telemetry(self, telemetry_coordinator) -> None:
        """Test health_check returns unhealthy when no account telemetry."""
        # Remove account telemetry from registry
        telemetry_coordinator.context.registry.extras = {}

        status = telemetry_coordinator.health_check()

        assert status.healthy is False
        assert status.component == "telemetry"
        assert status.details["has_account_telemetry"] is False

    def test_health_check_with_account_telemetry(
        self, telemetry_coordinator, mock_account_telemetry
    ) -> None:
        """Test health_check returns healthy when account telemetry is available."""
        telemetry_coordinator.context.registry.extras["account_telemetry"] = mock_account_telemetry

        status = telemetry_coordinator.health_check()

        assert status.healthy is True
        assert status.details["has_account_telemetry"] is True

    def test_should_enable_streaming_in_dev_profile(self, telemetry_coordinator) -> None:
        """Test _should_enable_streaming returns False in DEV profile."""
        telemetry_coordinator.context.config.perps_enable_streaming = True
        telemetry_coordinator.context.config.profile = Profile.DEV

        result = telemetry_coordinator._should_enable_streaming()

        assert result is False

    def test_should_enable_streaming_in_prod_profile(self, telemetry_coordinator) -> None:
        """Test _should_enable_streaming returns True in PROD profile with streaming enabled."""
        telemetry_coordinator.context.config.perps_enable_streaming = True
        telemetry_coordinator.context.config.profile = Profile.PROD

        result = telemetry_coordinator._should_enable_streaming()

        assert result is True

    def test_should_enable_streaming_when_disabled(self, telemetry_coordinator) -> None:
        """Test _should_enable_streaming returns False when streaming disabled."""
        telemetry_coordinator.context.config.perps_enable_streaming = False
        telemetry_coordinator.context.config.profile = Profile.PROD

        result = telemetry_coordinator._should_enable_streaming()

        assert result is False

    def test_extract_mark_from_message_with_bid_ask(self) -> None:
        """Test _extract_mark_from_message calculates mark from bid and ask."""
        msg = {"best_bid": "50000", "best_ask": "50100"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        assert result == Decimal("50050")

    def test_extract_mark_from_message_with_last_price(self) -> None:
        """Test _extract_mark_from_message extracts mark from last price."""
        msg = {"last": "50075"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        assert result == Decimal("50075")

    def test_extract_mark_from_message_with_price_field(self) -> None:
        """Test _extract_mark_from_message extracts mark from price field."""
        msg = {"price": "50025"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        assert result == Decimal("50025")

    def test_extract_mark_from_message_with_invalid_data(self) -> None:
        """Test _extract_mark_from_message handles negative price correctly."""
        # Test with negative price - it actually calculates the average
        msg = {"best_bid": "-100", "best_ask": "100"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        # The function calculates (-100 + 100) / 2 = 0
        assert result == Decimal("0")

    def test_extract_mark_from_message_with_non_numeric_data(self) -> None:
        """Test _extract_mark_from_message returns None for non-numeric data."""
        msg = {"best_bid": "invalid", "best_ask": "50100"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        assert result is None

    def test_extract_mark_from_message_with_no_data(self) -> None:
        """Test _extract_mark_from_message returns None for no price data."""
        msg = {"symbol": "BTC-PERP"}

        result = TelemetryCoordinator._extract_mark_from_message(msg)

        assert result is None

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
        telemetry_coordinator.context.registry.extras = {}

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

    def test_update_mark_and_metrics_emits_metric(self, telemetry_coordinator) -> None:
        """Test _update_mark_and_metrics emits mark update metric."""
        mark_price = Decimal("50000")

        with patch("bot_v2.orchestration.coordinators.telemetry.emit_metric") as mock_emit:
            telemetry_coordinator._update_mark_and_metrics(
                telemetry_coordinator.context, "BTC-PERP", mark_price
            )

            # Verify metric was emitted
            mock_emit.assert_called_once()
            call_args = mock_emit.call_args[0]
            assert call_args[2]["event_type"] == "ws_mark_update"
            assert call_args[2]["symbol"] == "BTC-PERP"
            assert call_args[2]["mark"] == "50000"

    def test_update_mark_and_metrics_updates_runtime_state_fallback(
        self, telemetry_coordinator
    ) -> None:
        """Test _update_mark_and_metrics falls back to runtime state when no strategy coordinator."""
        mark_price = Decimal("50000")

        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", mark_price
        )

        # Verify mark window updated in runtime state
        window = telemetry_coordinator.context.runtime_state.mark_windows["BTC-PERP"]
        assert len(window) == 1
        assert window[0] == mark_price

    def test_update_mark_and_metrics_handles_missing_strategy_coordinator(
        self, telemetry_coordinator
    ) -> None:
        """Test _update_mark_and_metrics handles missing strategy coordinator gracefully."""
        mark_price = Decimal("50000")

        # Should not raise exception even with no strategy coordinator
        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", mark_price
        )

    def test_update_mark_and_metrics_handles_missing_risk_manager(self) -> None:
        """Test _update_mark_and_metrics handles missing risk manager gracefully."""
        # Create a context without risk manager
        config = MagicMock()
        config.profile = Profile.DEV
        config.perps_enable_streaming = False
        config.account_telemetry_interval = 300
        config.short_ma = 10
        config.long_ma = 20

        registry = MagicMock()
        registry.extras = {}
        registry.risk_manager = None  # Explicitly set to None

        broker = MagicMock()
        event_store = MagicMock()
        runtime_state = MagicMock()
        runtime_state.mark_windows = {}
        runtime_state.mark_lock = MagicMock()

        from bot_v2.orchestration.coordinators.base import CoordinatorContext

        context_without_risk = CoordinatorContext(
            bot_id="test_bot",
            config=config,
            registry=registry,
            broker=broker,
            event_store=event_store,
            runtime_state=runtime_state,
            risk_manager=None,  # No risk manager
            symbols=[],
        )

        coordinator = TelemetryCoordinator(context=context_without_risk)
        mark_price = Decimal("50000")

        # Should not raise exception
        coordinator._update_mark_and_metrics(context_without_risk, "BTC-PERP", mark_price)

    def test_update_mark_and_metrics_handles_market_monitor_error(
        self, telemetry_coordinator, mock_market_monitor
    ) -> None:
        """Test _update_mark_and_metrics handles market monitor recording error."""
        mock_market_monitor.record_update.side_effect = Exception("Monitor error")
        telemetry_coordinator._market_monitor = mock_market_monitor
        mark_price = Decimal("50000")

        # Should not raise exception
        telemetry_coordinator._update_mark_and_metrics(
            telemetry_coordinator.context, "BTC-PERP", mark_price
        )

    async def test_start_background_tasks_with_no_services(self, telemetry_coordinator) -> None:
        """Test start_background_tasks returns empty list when no services available."""
        telemetry_coordinator.context.registry.extras = {}

        tasks = await telemetry_coordinator.start_background_tasks()

        assert tasks == []

    async def test_start_background_tasks_with_account_telemetry(
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

    async def test_shutdown_calls_parent_shutdown(self, telemetry_coordinator) -> None:
        """Test shutdown calls parent shutdown method."""
        with patch(
            "bot_v2.orchestration.coordinators.base.BaseCoordinator.shutdown",
            new_callable=AsyncMock,
        ) as mock_super_shutdown:
            await telemetry_coordinator.shutdown()

            mock_super_shutdown.assert_called_once()

    def test_init_market_services_updates_context(self, telemetry_coordinator) -> None:
        """Test init_market_services compatibility helper updates context."""
        original_context = telemetry_coordinator.context

        with patch.object(telemetry_coordinator, "initialize") as mock_init:
            mock_init.return_value = original_context

            telemetry_coordinator.init_market_services()

            mock_init.assert_called_once_with(original_context)

    async def test_start_streaming_background_early_return(self, telemetry_coordinator) -> None:
        """Test start_streaming_background returns early when streaming disabled."""
        telemetry_coordinator.context.config.perps_enable_streaming = False

        telemetry_coordinator.start_streaming_background()

        # Should not create any background tasks or raise exceptions
        assert len(telemetry_coordinator._background_tasks) == 0

    def test_stop_streaming_background_schedules_coroutine(self, telemetry_coordinator) -> None:
        """Test stop_streaming_background schedules stop coroutine."""
        with patch.object(telemetry_coordinator, "_schedule_coroutine") as mock_schedule:
            telemetry_coordinator.stop_streaming_background()

            mock_schedule.assert_called_once()
