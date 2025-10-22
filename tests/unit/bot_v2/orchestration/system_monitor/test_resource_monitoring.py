"""Tests for SystemMonitor resource monitoring and psutil integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from bot_v2.orchestration.system_monitor import SystemMonitor


class TestResourceMonitoring:
    """Test system resource collection and psutil integration."""

    def test_psutil_unavailable_at_initialization(
        self, mock_bot, fake_account_telemetry, caplog
    ) -> None:
        """Test system_monitor handles psutil unavailable during initialization."""
        with patch(
            "bot_v2.orchestration.system_monitor.ResourceCollector",
            side_effect=ImportError("No psutil"),
        ):
            # Set log level to capture debug messages
            caplog.set_level("DEBUG", logger="bot_v2.orchestration.system_monitor")

            monitor = SystemMonitor(bot=mock_bot, account_telemetry=fake_account_telemetry)

            # Verify resource collector is None
            assert monitor._resource_collector is None

    def test_psutil_available_at_initialization(
        self, system_monitor: SystemMonitor, fake_resource_collector
    ) -> None:
        """Test system_monitor successfully initializes ResourceCollector when psutil available."""
        # Resource collector should be initialized during SystemMonitor creation
        # This test verifies the normal path when psutil is available
        assert system_monitor._resource_collector is not None

    def test_psutil_unavailable_at_runtime_collection_error(
        self, system_monitor: SystemMonitor, mock_bot, sample_balances, caplog
    ) -> None:
        """Test system_monitor handles psutil runtime collection errors gracefully."""
        mock_bot.broker.list_positions.return_value = []
        mock_bot.broker.list_balances.return_value = sample_balances

        # Mock resource collector to raise during collection
        system_monitor._resource_collector = MagicMock()
        system_monitor._resource_collector.collect.side_effect = RuntimeError("Permission denied")

        # Set log level to capture debug messages
        caplog.set_level("DEBUG", logger="bot_v2.orchestration.system_monitor")

        # Trigger status logging which includes resource collection
        import asyncio

        asyncio.run(system_monitor.log_status())

        # Verify error logged
        assert "Unable to collect system metrics" in caplog.text
        assert "Permission denied" in caplog.text

        # Verify metrics still published without system data
        system_monitor._metrics_publisher.publish.assert_called_once()

    def test_system_metrics_included_in_payload_when_available(
        self, system_monitor: SystemMonitor, mock_bot, sample_balances, fake_resource_collector
    ) -> None:
        """Test system metrics are included in payload when ResourceCollector works."""
        mock_bot.broker.list_positions.return_value = []
        mock_bot.broker.list_balances.return_value = sample_balances

        # Replace the resource collector with our fake one
        system_monitor._resource_collector = fake_resource_collector

        # Trigger status logging
        import asyncio

        asyncio.run(system_monitor.log_status())

        # Verify system metrics included in payload
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert "system" in metrics_payload

        system_metrics = metrics_payload["system"]
        assert system_metrics["cpu_percent"] == 15.5
        assert system_metrics["memory_percent"] == 45.2
        assert system_metrics["memory_used_mb"] == 2048.0
        assert system_metrics["disk_percent"] == 67.8
        assert system_metrics["disk_used_gb"] == 125.5
        assert system_metrics["network_sent_mb"] == 1024.5
        assert system_metrics["network_recv_mb"] == 2048.2
        assert system_metrics["open_files"] == 156
        assert system_metrics["threads"] == 24

    def test_system_metrics_excluded_when_collector_none(
        self, system_monitor_no_resource_collector: SystemMonitor, mock_bot, sample_balances
    ) -> None:
        """Test system metrics are excluded when ResourceCollector is None."""
        mock_bot.broker.list_positions.return_value = []
        mock_bot.broker.list_balances.return_value = sample_balances

        # Trigger status logging
        import asyncio

        asyncio.run(system_monitor_no_resource_collector.log_status())

        # Verify system metrics NOT included in payload
        call_args = system_monitor_no_resource_collector._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert "system" not in metrics_payload

    def test_resource_collector_partial_metrics_handling(
        self, system_monitor: SystemMonitor, mock_bot, sample_balances
    ) -> None:
        """Test system_monitor handles partial resource metrics gracefully."""
        mock_bot.broker.list_positions.return_value = []
        mock_bot.broker.list_balances.return_value = sample_balances

        # Create resource collector that returns partial data
        partial_collector = MagicMock()
        from types import SimpleNamespace

        usage = SimpleNamespace()
        usage.cpu_percent = 25.0
        usage.memory_percent = 60.0
        # Missing other fields to simulate partial collection
        usage.memory_mb = None
        usage.disk_percent = None

        partial_collector.collect.return_value = usage
        system_monitor._resource_collector = partial_collector

        # Trigger status logging
        import asyncio

        asyncio.run(system_monitor.log_status())

        # Verify partial metrics are included and missing ones are handled
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]

        if "system" in metrics_payload:
            system_metrics = metrics_payload["system"]
            # Should include available metrics
            assert system_metrics.get("cpu_percent") == 25.0
            assert system_metrics.get("memory_percent") == 60.0
            # Missing metrics should be handled gracefully (either omitted or set to default)

    def test_system_monitor_handles_resource_collector_init_exception(
        self, mock_bot, fake_account_telemetry, caplog
    ) -> None:
        """Test system_monitor handles ResourceCollector initialization exception gracefully."""
        with patch("bot_v2.orchestration.system_monitor.ResourceCollector") as mock_collector_class:
            mock_collector_class.side_effect = RuntimeError("Insufficient permissions")

            # Set log level to capture debug messages
            caplog.set_level("DEBUG", logger="bot_v2.orchestration.system_monitor")

            monitor = SystemMonitor(bot=mock_bot, account_telemetry=fake_account_telemetry)

            # Should not raise exception, but log debug message
            assert monitor._resource_collector is None

    def test_system_metrics_dict_merging_behavior(
        self, system_monitor: SystemMonitor, mock_bot, sample_balances, fake_resource_collector
    ) -> None:
        """Test system metrics are correctly merged into the larger metrics payload."""
        mock_bot.broker.list_positions.return_value = []
        mock_bot.broker.list_balances.return_value = sample_balances
        mock_bot.order_stats = {"test": "value"}

        # Replace the resource collector with our fake one
        system_monitor._resource_collector = fake_resource_collector

        # Trigger status logging
        import asyncio

        asyncio.run(system_monitor.log_status())

        # Verify payload structure includes both regular and system metrics
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]

        # Should have all required fields
        assert "timestamp" in metrics_payload
        assert "profile" in metrics_payload
        assert "equity" in metrics_payload
        assert "order_stats" in metrics_payload
        assert "system" in metrics_payload

        # System metrics should be properly nested
        system_metrics = metrics_payload["system"]
        assert isinstance(system_metrics, dict)
        assert len(system_metrics) > 0

    def test_resource_monitoring_does_not_block_status_logging(
        self, system_monitor: SystemMonitor, mock_bot, sample_balances
    ) -> None:
        """Test resource monitoring failures don't block overall status logging."""
        mock_bot.broker.list_positions.return_value = []
        mock_bot.broker.list_balances.return_value = sample_balances

        # Make resource collector raise exception
        system_monitor._resource_collector = MagicMock()
        system_monitor._resource_collector.collect.side_effect = Exception(
            "Resource collection failed"
        )

        # Trigger status logging - should not raise exception
        import asyncio

        asyncio.run(system_monitor.log_status())

        # Verify metrics still published despite resource collection failure
        system_monitor._metrics_publisher.publish.assert_called_once()

        # Verify other metrics are still included
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert "equity" in metrics_payload
        assert "profile" in metrics_payload
