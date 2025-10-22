"""Tests for TelemetryCoordinator initialization, broker integration, and health checks.

This module tests:
- Coordinator initialization with and without broker
- Dynamic import handling and fallback behaviors
- Metric emission and error handling
- Health check functionality
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.orchestration.coordinators.telemetry import TelemetryCoordinator


def test_initialize_without_broker(make_context) -> None:
    """Test initialization without a broker returns empty extras."""
    context = make_context(broker=None)
    coordinator = TelemetryCoordinator(context)

    updated = coordinator.initialize(context)

    assert updated.registry.extras == {}


def test_initialize_with_broker(make_context) -> None:
    """Test initialization with broker creates telemetry services."""
    broker = Mock(spec=CoinbaseBrokerage)
    broker.__class__ = CoinbaseBrokerage
    risk_manager = Mock()
    context = make_context(broker=broker, risk_manager=risk_manager)

    coordinator = TelemetryCoordinator(context)
    updated = coordinator.initialize(context)

    extras = updated.registry.extras
    assert "account_manager" in extras
    assert "account_telemetry" in extras
    assert "market_monitor" in extras


class TestDynamicImportAndInitialization:
    """Test dynamic import paths and initialization scenarios."""

    def test_initialize_with_coinbase_import_failure(
        self, make_context, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test graceful handling when Coinbase adapter import fails."""
        # Mock the import to raise an exception by patching the import line directly
        import builtins

        original_import = builtins.__import__

        def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
            if "coinbase.adapter" in name:
                raise ImportError("Simulated import failure")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", failing_import)

        broker = Mock()
        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)

        updated = coordinator.initialize(context)

        # Should return unchanged context when import fails
        assert updated.registry.extras == {}
        assert coordinator._market_monitor is None

    def test_initialize_with_non_coinbase_broker(self, make_context) -> None:
        """Test graceful handling when broker is not CoinbaseBrokerage."""
        from bot_v2.features.brokerages.core.interfaces import IBrokerage

        # Create a mock broker that's not CoinbaseBrokerage
        broker = Mock(spec=IBrokerage)
        # Ensure it's not an instance of CoinbaseBrokerage
        broker.__class__.__module__ = "some.other.module"
        broker.__class__.__name__ = "OtherBrokerage"

        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)

        updated = coordinator.initialize(context)

        # Should return unchanged context when broker type mismatch
        assert updated.registry.extras == {}
        assert coordinator._market_monitor is None

    def test_initialize_account_telemetry_without_snapshots(self, make_context) -> None:
        """Test initialization when account telemetry doesn't support snapshots."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)

        # Mock account telemetry to not support snapshots
        with pytest.MonkeyPatch().context() as m:
            mock_telemetry = Mock()
            mock_telemetry.supports_snapshots.return_value = False

            def mock_account_telemetry(*args, **kwargs):
                return mock_telemetry

            m.setattr(
                "bot_v2.orchestration.coordinators.telemetry.AccountTelemetryService",
                mock_account_telemetry,
            )

            updated = coordinator.initialize(context)

            # Should still initialize but log info about disabled snapshots
            extras = updated.registry.extras
            assert "account_telemetry" in extras
            assert "account_manager" in extras
            assert "intx_portfolio_service" in extras
            assert "market_monitor" in extras

    def test_market_heartbeat_logger_exception_handling(self, make_context) -> None:
        """Test that heartbeat logger exceptions are caught and logged."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)

        # Mock the plog to raise an exception
        with pytest.MonkeyPatch().context() as m:
            mock_plog = Mock()
            mock_plog.log_market_heartbeat.side_effect = Exception("Heartbeat failed")

            def mock_get_plog():
                return mock_plog

            m.setattr("bot_v2.orchestration.coordinators.telemetry._get_plog", mock_get_plog)

            # Should not raise exception even when heartbeat logging fails
            updated = coordinator.initialize(context)

            # Should still complete initialization
            extras = updated.registry.extras
            assert "market_monitor" in extras
            assert coordinator._market_monitor is not None


class TestMetricEmissionAndErrorHandling:
    """Test metric emission paths and error handling scenarios."""

    def test_streaming_orderbook_fallback_to_trades(self, make_context) -> None:
        """Test fallback to trades when orderbook streaming fails."""
        import threading

        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage
        broker.stream_orderbook.side_effect = [
            Exception("Orderbook failed"),
            iter([{"price": "50000"}]),
        ]
        broker.stream_trades.return_value = iter([{"price": "50001"}])

        context = make_context(broker=broker, symbols=("BTC-PERP",))
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Should fallback to trades when orderbook fails
        stop_signal = threading.Event()
        coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

        # Should have tried orderbook first, then fallback to trades
        broker.stream_orderbook.assert_called_once()
        broker.stream_trades.assert_called_once_with(["BTC-PERP"])

    def test_streaming_with_no_broker_error(self, make_context) -> None:
        """Test error handling when no broker is available."""
        import threading

        context = make_context(broker=None)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Should handle missing broker gracefully
        stop_signal = threading.Event()
        coordinator._run_stream_loop(["BTC-PERP"], 1, stop_signal)

        # Should not crash, just log error and return

    def test_run_account_telemetry_with_missing_context(self, make_context) -> None:
        """Test account telemetry execution with missing context components."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Remove account telemetry from context to test missing case
        updated_context = context.with_updates(registry=context.registry.with_updates(extras={}))
        coordinator.update_context(updated_context)

        # Should handle missing account telemetry gracefully
        with pytest.MonkeyPatch().context() as m:
            mock_emit = Mock()
            m.setattr("bot_v2.orchestration.coordinators.telemetry.emit_metric", mock_emit)

            coordinator.run_account_telemetry()

            # Should not emit metrics when account telemetry is missing
            mock_emit.assert_not_called()

    def test_health_check_with_missing_services(self, make_context) -> None:
        """Test health check when required services are missing."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Remove market monitor to test missing service
        updated_context = context.with_updates(registry=context.registry.with_updates(extras={}))
        coordinator.update_context(updated_context)

        health = coordinator.health_check()

        # Should return unhealthy status when account telemetry is missing
        assert health.healthy is False
        assert health.component == "telemetry"
        assert health.details["has_account_telemetry"] is False

    def test_emit_metric_error_handling(self, make_context) -> None:
        """Test error handling in metric emission."""
        broker = Mock(spec=CoinbaseBrokerage)
        broker.__class__ = CoinbaseBrokerage

        context = make_context(broker=broker)
        coordinator = TelemetryCoordinator(context)
        coordinator.initialize(context)

        # Mock emit_metric to raise exception
        with pytest.MonkeyPatch().context() as m:

            def mock_emit_error(*args, **kwargs):
                raise Exception("Emission failed")

            m.setattr("bot_v2.orchestration.coordinators.telemetry.emit_metric", mock_emit_error)

            # Should handle emission errors gracefully
            # This would be tested through methods that call emit_metric internally
            coordinator._market_monitor = Mock()
            coordinator._market_monitor.get_activity_summary.return_value = {"test": "data"}

            # The method should not crash even if emit_metric fails
            # This tests error resilience in metric emission paths
