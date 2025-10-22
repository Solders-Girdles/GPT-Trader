"""Tests for SystemMonitor status logging and health integration functionality."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from bot_v2.orchestration.system_monitor import SystemMonitor


class TestStatusLogging:
    """Test async status logging and metrics payload construction."""

    async def test_log_status_fetches_positions_successfully(
        self, system_monitor: SystemMonitor, mock_bot, sample_positions
    ) -> None:
        """Test log_status successfully fetches and processes positions."""
        mock_bot.broker.list_positions.return_value = sample_positions
        mock_bot.broker.list_balances.return_value = []

        await system_monitor.log_status()

        # Verify broker was called
        mock_bot.broker.list_positions.assert_called_once()
        system_monitor._metrics_publisher.publish.assert_called_once()

        # Check metrics payload includes positions
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert len(metrics_payload["positions"]) == 2
        assert metrics_payload["positions"][0]["symbol"] == "BTC-PERP"
        assert metrics_payload["positions"][0]["quantity"] == 0.5

    async def test_log_status_handles_position_fetch_error(
        self, system_monitor: SystemMonitor, mock_bot, caplog
    ) -> None:
        """Test log_status handles position fetch errors gracefully."""
        mock_bot.broker.list_positions.side_effect = Exception("Network error")
        mock_bot.broker.list_balances.return_value = []

        # Set log level to capture warnings
        caplog.set_level("WARNING", logger="bot_v2.orchestration.system_monitor")

        await system_monitor.log_status()

        # Verify error was logged
        assert "Unable to fetch positions for status log" in caplog.text
        assert "Network error" in caplog.text

        # Verify metrics still published with empty positions
        system_monitor._metrics_publisher.publish.assert_called_once()
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert metrics_payload["positions"] == []

    async def test_log_status_handles_balance_fetch_error(
        self, system_monitor: SystemMonitor, mock_bot, sample_positions, caplog
    ) -> None:
        """Test log_status handles balance fetch errors gracefully."""
        mock_bot.broker.list_positions.return_value = sample_positions
        mock_bot.broker.list_balances.side_effect = Exception("Balance error")

        # Set log level to capture warnings
        caplog.set_level("WARNING", logger="bot_v2.orchestration.system_monitor")

        await system_monitor.log_status()

        # Verify error was logged
        assert "Unable to fetch balances for status log" in caplog.text
        assert "Balance error" in caplog.text

        # Verify equity defaults to 0 when no USD balance
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert metrics_payload["equity"] == 0.0

    async def test_log_status_calculates_equity_from_usd_balance(
        self, system_monitor: SystemMonitor, mock_bot, sample_positions, sample_balances
    ) -> None:
        """Test log_status correctly calculates equity from USD balance."""
        mock_bot.broker.list_positions.return_value = sample_positions
        mock_bot.broker.list_balances.return_value = sample_balances

        await system_monitor.log_status()

        # Verify equity calculation
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert metrics_payload["equity"] == 8500.0

    async def test_log_status_defaults_equity_to_zero_when_no_usd_balance(
        self, system_monitor: SystemMonitor, mock_bot, sample_positions
    ) -> None:
        """Test log_status defaults equity to 0 when no USD balance found."""
        mock_bot.broker.list_positions.return_value = sample_positions
        mock_bot.broker.list_balances.return_value = [
            MagicMock(asset="BTC", available=Decimal("0.1"), total=Decimal("0.1"))
        ]

        await system_monitor.log_status()

        # Verify equity defaults to 0
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert metrics_payload["equity"] == 0.0

    async def test_log_status_formats_status_banner_and_summary(
        self, system_monitor: SystemMonitor, mock_bot, sample_positions, sample_balances, caplog
    ) -> None:
        """Test log_status formats status banner and summary correctly."""
        mock_bot.broker.list_positions.return_value = sample_positions
        mock_bot.broker.list_balances.return_value = sample_balances

        # Set log level to capture info messages
        caplog.set_level("INFO", logger="bot_v2.orchestration.system_monitor")

        await system_monitor.log_status()

        # Verify banner and summary logged
        log_messages = [
            record.message for record in caplog.records if "Banner" not in record.message
        ]
        assert any("=" * 60 in msg for msg in log_messages)
        assert any(
            "Bot Status" in msg and "Equity: $8500" in msg and "Positions: 2" in msg
            for msg in log_messages
        )

    async def test_log_status_logs_decisions_iteration(
        self, system_monitor: SystemMonitor, mock_bot, sample_balances, sample_decisions, caplog
    ) -> None:
        """Test log_status iterates through and logs all decisions."""
        mock_bot.broker.list_positions.return_value = []
        mock_bot.broker.list_balances.return_value = sample_balances
        mock_bot.last_decisions = sample_decisions

        # Set log level to capture info messages
        caplog.set_level("INFO", logger="bot_v2.orchestration.system_monitor")

        await system_monitor.log_status()

        # Verify decisions logged
        log_messages = [record.message for record in caplog.records]
        decision_logs = [msg for msg in log_messages if "BTC-PERP:" in msg or "ETH-PERP:" in msg]
        assert len(decision_logs) == 2
        assert any("BTC-PERP: HOLD" in msg for msg in decision_logs)
        assert any("ETH-PERP: BUY" in msg for msg in decision_logs)

    async def test_log_status_handles_open_orders_database_error(
        self, system_monitor: SystemMonitor, mock_bot, sample_balances
    ) -> None:
        """Test log_status handles open orders database error gracefully."""
        mock_bot.broker.list_positions.return_value = []
        mock_bot.broker.list_balances.return_value = sample_balances
        mock_bot.orders_store.get_open_orders.side_effect = Exception("Database error")

        await system_monitor.log_status()

        # Verify open_orders defaults to 0 on error
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert metrics_payload["open_orders"] == 0

    async def test_log_status_constructs_full_metrics_payload(
        self,
        system_monitor: SystemMonitor,
        mock_bot,
        sample_positions,
        sample_balances,
        sample_decisions,
    ) -> None:
        """Test log_status constructs complete metrics payload with all fields."""
        mock_bot.broker.list_positions.return_value = sample_positions
        mock_bot.broker.list_balances.return_value = sample_balances
        mock_bot.last_decisions = sample_decisions
        mock_bot.orders_store.get_open_orders.return_value = [MagicMock(), MagicMock()]
        mock_bot.order_stats = {"attempted": 10, "successful": 8, "failed": 2}

        await system_monitor.log_status()

        # Verify complete payload structure
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]

        # Check all required fields
        assert "timestamp" in metrics_payload
        assert "profile" in metrics_payload
        assert "equity" in metrics_payload
        assert "positions" in metrics_payload
        assert "decisions" in metrics_payload
        assert "order_stats" in metrics_payload
        assert "open_orders" in metrics_payload
        assert "uptime_seconds" in metrics_payload

        # Check specific values
        assert metrics_payload["profile"] == "prod"
        assert metrics_payload["equity"] == 8500.0
        assert len(metrics_payload["positions"]) == 2
        assert len(metrics_payload["decisions"]) == 2
        assert metrics_payload["order_stats"] == {"attempted": 10, "successful": 8, "failed": 2}
        assert metrics_payload["open_orders"] == 2

    async def test_log_status_handles_positions_with_bad_data(
        self, system_monitor: SystemMonitor, mock_bot, sample_balances, caplog
    ) -> None:
        """Test log_status handles positions with bad data gracefully."""
        # Create positions with bad data
        good_pos = MagicMock(symbol="BTC-PERP", quantity=Decimal("0.5"), side="long")
        bad_pos = MagicMock(symbol="ETH-PERP")  # Missing required attributes

        mock_bot.broker.list_positions.return_value = [good_pos, bad_pos]
        mock_bot.broker.list_balances.return_value = sample_balances

        # Mock quantity_from to raise for bad position
        with patch("bot_v2.orchestration.system_monitor.quantity_from") as mock_quantity:

            def quantity_side_effect(pos):
                if pos == bad_pos:
                    raise ValueError("Invalid position data")
                return Decimal("0.5")

            mock_quantity.side_effect = quantity_side_effect

            # Set log level to capture debug messages
            caplog.set_level("DEBUG", logger="bot_v2.orchestration.system_monitor")

            await system_monitor.log_status()

            # Should still publish metrics with good position only
            system_monitor._metrics_publisher.publish.assert_called_once()
            call_args = system_monitor._metrics_publisher.publish.call_args
            metrics_payload = call_args[0][0]
            assert len(metrics_payload["positions"]) == 1
            assert metrics_payload["positions"][0]["symbol"] == "BTC-PERP"

    async def test_log_status_includes_account_telemetry_snapshot(
        self, system_monitor: SystemMonitor, mock_bot, sample_positions, sample_balances
    ) -> None:
        """Test log_status includes account telemetry snapshot when available."""
        mock_bot.broker.list_positions.return_value = sample_positions
        mock_bot.broker.list_balances.return_value = sample_balances

        await system_monitor.log_status()

        # Verify account snapshot included
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert "account_snapshot" in metrics_payload
        assert metrics_payload["account_snapshot"]["total_value"] == 10000

    async def test_log_status_handles_missing_account_telemetry(
        self, system_monitor: SystemMonitor, mock_bot, sample_positions, sample_balances
    ) -> None:
        """Test log_status handles missing account telemetry gracefully."""
        mock_bot.broker.list_positions.return_value = sample_positions
        mock_bot.broker.list_balances.return_value = sample_balances
        system_monitor._account_telemetry = None

        await system_monitor.log_status()

        # Verify account snapshot not included
        call_args = system_monitor._metrics_publisher.publish.call_args
        metrics_payload = call_args[0][0]
        assert "account_snapshot" not in metrics_payload

    def test_write_health_status_delegates_to_metrics_publisher(
        self, system_monitor: SystemMonitor, fake_metrics_publisher
    ) -> None:
        """Test write_health_status correctly delegates to metrics publisher."""
        system_monitor.write_health_status(ok=True, message="All systems operational")

        fake_metrics_publisher.write_health_status.assert_called_once_with(
            ok=True, message="All systems operational", error=""
        )

    def test_write_health_status_with_error_state(
        self, system_monitor: SystemMonitor, fake_metrics_publisher
    ) -> None:
        """Test write_health_status correctly passes error state."""
        system_monitor.write_health_status(ok=False, error="Connection timeout")

        fake_metrics_publisher.write_health_status.assert_called_once_with(
            ok=False, message="", error="Connection timeout"
        )

    async def test_run_position_reconciliation_delegates_to_position_reconciler(
        self, system_monitor: SystemMonitor, mock_bot, fake_position_reconciler
    ) -> None:
        """Test run_position_reconciliation correctly delegates to position reconciler."""
        await system_monitor.run_position_reconciliation(interval_seconds=60)

        fake_position_reconciler.run.assert_called_once_with(mock_bot, interval_seconds=60)
