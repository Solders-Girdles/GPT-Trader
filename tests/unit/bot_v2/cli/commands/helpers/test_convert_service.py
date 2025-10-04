"""Tests for convert service."""

import json
from unittest.mock import Mock

import pytest

from bot_v2.cli.commands.convert_request_parser import ConvertRequest
from bot_v2.cli.commands.convert_service import ConvertService


class TestConvertService:
    """Test ConvertService."""

    def test_execute_conversion_success(self):
        """Test successful conversion execution."""
        bot = Mock()
        bot.account_manager.convert.return_value = {
            "conversion_id": "conv-123",
            "from": "USD",
            "to": "BTC",
            "amount": "1000",
            "status": "completed",
        }
        request = ConvertRequest(from_asset="USD", to_asset="BTC", amount="1000")

        printed_output = []
        service = ConvertService(printer=printed_output.append)
        result = service.execute_conversion(bot, request)

        assert result == 0
        bot.account_manager.convert.assert_called_once_with(
            {"from": "USD", "to": "BTC", "amount": "1000"}, commit=True
        )
        assert len(printed_output) == 1

        # Verify JSON output
        parsed = json.loads(printed_output[0])
        assert parsed["conversion_id"] == "conv-123"
        assert parsed["from"] == "USD"
        assert parsed["to"] == "BTC"

    def test_execute_conversion_with_custom_printer(self):
        """Test service uses custom printer when provided."""
        bot = Mock()
        bot.account_manager.convert.return_value = {"result": "success"}
        request = ConvertRequest(from_asset="USD", to_asset="BTC", amount="1000")

        mock_printer = Mock()
        service = ConvertService(printer=mock_printer)
        service.execute_conversion(bot, request)

        mock_printer.assert_called_once()
        # Verify printer received JSON string
        call_args = mock_printer.call_args[0][0]
        parsed = json.loads(call_args)
        assert parsed["result"] == "success"

    def test_execute_conversion_commit_true(self):
        """Test conversion called with commit=True."""
        bot = Mock()
        bot.account_manager.convert.return_value = {}
        request = ConvertRequest(from_asset="USD", to_asset="BTC", amount="1000")

        service = ConvertService()
        service.execute_conversion(bot, request)

        # Verify commit=True is passed
        call_args = bot.account_manager.convert.call_args
        assert call_args[1]["commit"] is True

    def test_execute_conversion_exception_propagates(self):
        """Test that conversion exceptions are propagated."""
        bot = Mock()
        bot.account_manager.convert.side_effect = Exception("API error")
        request = ConvertRequest(from_asset="USD", to_asset="BTC", amount="1000")

        service = ConvertService()

        with pytest.raises(Exception, match="API error"):
            service.execute_conversion(bot, request)

    def test_json_formatting_with_indent(self):
        """Test JSON output is formatted with indentation."""
        bot = Mock()
        bot.account_manager.convert.return_value = {
            "key": "value",
            "nested": {"data": "test"},
        }
        request = ConvertRequest(from_asset="USD", to_asset="BTC", amount="1000")

        printed_output = []
        service = ConvertService(printer=printed_output.append)
        service.execute_conversion(bot, request)

        output = printed_output[0]
        # Should have indentation (newlines and spaces)
        assert "\n" in output
        assert "  " in output

    def test_json_handles_special_types(self):
        """Test JSON formatting handles non-serializable types."""
        from datetime import datetime
        from decimal import Decimal

        bot = Mock()
        bot.account_manager.convert.return_value = {
            "amount": Decimal("1000.50"),
            "timestamp": datetime(2023, 1, 1, 12, 0, 0),
        }
        request = ConvertRequest(from_asset="USD", to_asset="BTC", amount="1000")

        printed_output = []
        service = ConvertService(printer=printed_output.append)
        result = service.execute_conversion(bot, request)

        # Should not crash
        assert result == 0
        assert len(printed_output) == 1

    def test_payload_built_from_request(self):
        """Test payload correctly built from ConvertRequest."""
        bot = Mock()
        bot.account_manager.convert.return_value = {}
        request = ConvertRequest(from_asset="ETH", to_asset="USDC", amount="10.5")

        service = ConvertService()
        service.execute_conversion(bot, request)

        bot.account_manager.convert.assert_called_once_with(
            {"from": "ETH", "to": "USDC", "amount": "10.5"}, commit=True
        )
