"""Tests for account snapshot service."""

import json
from unittest.mock import Mock

import pytest

from bot_v2.cli.commands.account_snapshot_service import AccountSnapshotService


class TestAccountSnapshotService:
    """Test AccountSnapshotService."""

    def test_collect_and_print_success(self):
        """Test successful snapshot collection and printing."""
        bot = Mock()
        telemetry = Mock()
        telemetry.supports_snapshots.return_value = True
        telemetry.collect_snapshot.return_value = {
            "account_id": "acc-123",
            "balance": "10000.00",
        }
        bot.account_telemetry = telemetry

        printed_output = []
        service = AccountSnapshotService(printer=printed_output.append)
        result = service.collect_and_print(bot)

        assert result == 0
        telemetry.supports_snapshots.assert_called_once()
        telemetry.collect_snapshot.assert_called_once()
        assert len(printed_output) == 1

        # Verify JSON output
        parsed = json.loads(printed_output[0])
        assert parsed["account_id"] == "acc-123"
        assert parsed["balance"] == "10000.00"

    def test_collect_and_print_with_custom_printer(self):
        """Test service uses custom printer when provided."""
        bot = Mock()
        telemetry = Mock()
        telemetry.supports_snapshots.return_value = True
        telemetry.collect_snapshot.return_value = {"data": "test"}
        bot.account_telemetry = telemetry

        mock_printer = Mock()
        service = AccountSnapshotService(printer=mock_printer)
        service.collect_and_print(bot)

        mock_printer.assert_called_once()
        # Verify printer received JSON string
        call_args = mock_printer.call_args[0][0]
        parsed = json.loads(call_args)
        assert parsed["data"] == "test"

    def test_no_telemetry_raises_runtime_error(self):
        """Test RuntimeError when bot has no telemetry."""
        bot = Mock()
        bot.account_telemetry = None

        service = AccountSnapshotService()

        with pytest.raises(RuntimeError, match="Account snapshot telemetry is not available"):
            service.collect_and_print(bot)

    def test_telemetry_not_supported_raises_runtime_error(self):
        """Test RuntimeError when telemetry doesn't support snapshots."""
        bot = Mock()
        telemetry = Mock()
        telemetry.supports_snapshots.return_value = False
        bot.account_telemetry = telemetry

        service = AccountSnapshotService()

        with pytest.raises(RuntimeError, match="Account snapshot telemetry is not available"):
            service.collect_and_print(bot)

        telemetry.supports_snapshots.assert_called_once()

    def test_collection_exception_propagates(self):
        """Test that collection exceptions are propagated."""
        bot = Mock()
        telemetry = Mock()
        telemetry.supports_snapshots.return_value = True
        telemetry.collect_snapshot.side_effect = Exception("API error")
        bot.account_telemetry = telemetry

        service = AccountSnapshotService()

        with pytest.raises(Exception, match="API error"):
            service.collect_and_print(bot)

    def test_json_formatting_with_indent(self):
        """Test JSON output is formatted with indentation."""
        bot = Mock()
        telemetry = Mock()
        telemetry.supports_snapshots.return_value = True
        telemetry.collect_snapshot.return_value = {"key": "value", "nested": {"data": "test"}}
        bot.account_telemetry = telemetry

        printed_output = []
        service = AccountSnapshotService(printer=printed_output.append)
        service.collect_and_print(bot)

        output = printed_output[0]
        # Should have indentation (newlines and spaces)
        assert "\n" in output
        assert "  " in output

    def test_json_handles_special_types(self):
        """Test JSON formatting handles non-serializable types."""
        from datetime import datetime
        from decimal import Decimal

        bot = Mock()
        telemetry = Mock()
        telemetry.supports_snapshots.return_value = True
        telemetry.collect_snapshot.return_value = {
            "balance": Decimal("10000.50"),
            "timestamp": datetime(2023, 1, 1, 12, 0, 0),
        }
        bot.account_telemetry = telemetry

        printed_output = []
        service = AccountSnapshotService(printer=printed_output.append)
        result = service.collect_and_print(bot)

        # Should not crash
        assert result == 0
        assert len(printed_output) == 1
