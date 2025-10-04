"""Tests for move funds service."""

import json
from unittest.mock import Mock

import pytest

from bot_v2.cli.commands.move_funds_request_parser import MoveFundsRequest
from bot_v2.cli.commands.move_funds_service import MoveFundsService


class TestMoveFundsService:
    """Test MoveFundsService."""

    def test_execute_fund_movement_success(self):
        """Test successful fund movement execution."""
        bot = Mock()
        bot.account_manager.move_funds.return_value = {
            "transfer_id": "xfer-456",
            "from_portfolio": "port-123",
            "to_portfolio": "port-789",
            "amount": "5000",
            "status": "completed",
        }
        request = MoveFundsRequest(
            from_portfolio="port-123", to_portfolio="port-789", amount="5000"
        )

        printed_output = []
        service = MoveFundsService(printer=printed_output.append)
        result = service.execute_fund_movement(bot, request)

        assert result == 0
        bot.account_manager.move_funds.assert_called_once_with(
            {"from_portfolio": "port-123", "to_portfolio": "port-789", "amount": "5000"}
        )
        assert len(printed_output) == 1

        # Verify JSON output
        parsed = json.loads(printed_output[0])
        assert parsed["transfer_id"] == "xfer-456"
        assert parsed["from_portfolio"] == "port-123"
        assert parsed["to_portfolio"] == "port-789"

    def test_execute_fund_movement_with_custom_printer(self):
        """Test service uses custom printer when provided."""
        bot = Mock()
        bot.account_manager.move_funds.return_value = {"result": "success"}
        request = MoveFundsRequest(
            from_portfolio="port-123", to_portfolio="port-789", amount="5000"
        )

        mock_printer = Mock()
        service = MoveFundsService(printer=mock_printer)
        service.execute_fund_movement(bot, request)

        mock_printer.assert_called_once()
        # Verify printer received JSON string
        call_args = mock_printer.call_args[0][0]
        parsed = json.loads(call_args)
        assert parsed["result"] == "success"

    def test_execute_fund_movement_no_commit_flag(self):
        """Test fund movement called without commit flag (unlike convert)."""
        bot = Mock()
        bot.account_manager.move_funds.return_value = {}
        request = MoveFundsRequest(
            from_portfolio="port-123", to_portfolio="port-789", amount="5000"
        )

        service = MoveFundsService()
        service.execute_fund_movement(bot, request)

        # Verify NO commit flag is passed (single positional arg: payload)
        call_args = bot.account_manager.move_funds.call_args
        assert len(call_args[0]) == 1  # Only payload, no commit arg
        assert len(call_args[1]) == 0  # No keyword args

    def test_execute_fund_movement_exception_propagates(self):
        """Test that fund movement exceptions are propagated."""
        bot = Mock()
        bot.account_manager.move_funds.side_effect = Exception("Insufficient balance")
        request = MoveFundsRequest(
            from_portfolio="port-123", to_portfolio="port-789", amount="5000"
        )

        service = MoveFundsService()

        with pytest.raises(Exception, match="Insufficient balance"):
            service.execute_fund_movement(bot, request)

    def test_json_formatting_with_indent(self):
        """Test JSON output is formatted with indentation."""
        bot = Mock()
        bot.account_manager.move_funds.return_value = {
            "key": "value",
            "nested": {"data": "test"},
        }
        request = MoveFundsRequest(
            from_portfolio="port-123", to_portfolio="port-789", amount="5000"
        )

        printed_output = []
        service = MoveFundsService(printer=printed_output.append)
        service.execute_fund_movement(bot, request)

        output = printed_output[0]
        # Should have indentation (newlines and spaces)
        assert "\n" in output
        assert "  " in output

    def test_json_handles_special_types(self):
        """Test JSON formatting handles non-serializable types."""
        from datetime import datetime
        from decimal import Decimal

        bot = Mock()
        bot.account_manager.move_funds.return_value = {
            "amount": Decimal("5000.50"),
            "timestamp": datetime(2023, 1, 1, 12, 0, 0),
        }
        request = MoveFundsRequest(
            from_portfolio="port-123", to_portfolio="port-789", amount="5000"
        )

        printed_output = []
        service = MoveFundsService(printer=printed_output.append)
        result = service.execute_fund_movement(bot, request)

        # Should not crash
        assert result == 0
        assert len(printed_output) == 1

    def test_payload_built_from_request(self):
        """Test payload correctly built from MoveFundsRequest."""
        bot = Mock()
        bot.account_manager.move_funds.return_value = {}
        request = MoveFundsRequest(
            from_portfolio="port-aaa", to_portfolio="port-bbb", amount="2500.75"
        )

        service = MoveFundsService()
        service.execute_fund_movement(bot, request)

        bot.account_manager.move_funds.assert_called_once_with(
            {"from_portfolio": "port-aaa", "to_portfolio": "port-bbb", "amount": "2500.75"}
        )
