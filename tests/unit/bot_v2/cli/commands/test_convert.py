"""Tests for convert CLI command."""

import argparse
import json
from unittest.mock import Mock, patch

import pytest

from bot_v2.cli.commands.convert import handle_convert


@pytest.fixture
def mock_bot_with_account_manager():
    """Create a mock PerpsBot with account manager."""
    bot = Mock()
    account_mgr = Mock()
    account_mgr.convert.return_value = {
        "conversion_id": "conv-123",
        "from": "USD",
        "to": "BTC",
        "amount": "1000.00",
        "result": "0.02",
        "status": "completed",
    }
    bot.account_manager = account_mgr
    return bot


@pytest.fixture
def mock_parser():
    """Create a mock ArgumentParser."""
    parser = Mock(spec=argparse.ArgumentParser)
    return parser


class TestHandleConvert:
    """Tests for handle_convert function."""

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_successful_conversion(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser, capsys
    ):
        """Test successful asset conversion."""
        result = handle_convert("USD:BTC:1000", mock_bot_with_account_manager, mock_parser)

        assert result == 0
        mock_bot_with_account_manager.account_manager.convert.assert_called_once_with(
            {"from": "USD", "to": "BTC", "amount": "1000"}, commit=True
        )
        mock_shutdown.assert_called_once_with(mock_bot_with_account_manager)

        # Check output
        captured = capsys.readouterr()
        assert "conv-123" in captured.out
        assert "USD" in captured.out
        assert "BTC" in captured.out

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_conversion_output_is_json(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser, capsys
    ):
        """Test that conversion output is valid JSON."""
        handle_convert("USD:BTC:1000", mock_bot_with_account_manager, mock_parser)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "conversion_id" in parsed
        assert parsed["from"] == "USD"
        assert parsed["to"] == "BTC"

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_invalid_format_missing_parts(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser, caplog
    ):
        """Test error when conversion format is invalid (missing parts)."""
        import logging

        mock_parser.error.side_effect = SystemExit(2)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                handle_convert("USD:BTC", mock_bot_with_account_manager, mock_parser)

        # Verify error message is logged
        assert "Invalid convert argument format" in caplog.text
        assert "USD:BTC" in caplog.text
        mock_parser.error.assert_called_once_with("--convert requires format FROM:TO:AMOUNT")

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_invalid_format_single_value(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser
    ):
        """Test error when conversion format has no separators."""
        mock_parser.error.side_effect = SystemExit(2)

        with pytest.raises(SystemExit):
            handle_convert("INVALID", mock_bot_with_account_manager, mock_parser)

        mock_parser.error.assert_called_once()

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_conversion_with_whitespace(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser
    ):
        """Test conversion with whitespace around values."""
        result = handle_convert(" USD : BTC : 1000 ", mock_bot_with_account_manager, mock_parser)

        assert result == 0
        mock_bot_with_account_manager.account_manager.convert.assert_called_once_with(
            {"from": "USD", "to": "BTC", "amount": "1000"}, commit=True
        )

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_conversion_exception_raised(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser, caplog
    ):
        """Test exception during conversion is raised and logged."""
        import logging

        mock_bot_with_account_manager.account_manager.convert.side_effect = Exception(
            "Conversion failed"
        )

        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception, match="Conversion failed"):
                handle_convert("USD:BTC:1000", mock_bot_with_account_manager, mock_parser)

        # Verify error was logged
        assert "Conversion failed" in caplog.text
        # Shutdown should still be called in finally block
        mock_shutdown.assert_called_once_with(mock_bot_with_account_manager)

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_conversion_with_decimal_amount(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser
    ):
        """Test conversion with decimal amount."""
        mock_bot_with_account_manager.account_manager.convert.return_value = {
            "from": "BTC",
            "to": "ETH",
            "amount": "0.5",
            "result": "8.5",
        }

        result = handle_convert("BTC:ETH:0.5", mock_bot_with_account_manager, mock_parser)

        assert result == 0
        mock_bot_with_account_manager.account_manager.convert.assert_called_once_with(
            {"from": "BTC", "to": "ETH", "amount": "0.5"}, commit=True
        )

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_conversion_commit_flag(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser
    ):
        """Test that conversion is called with commit=True."""
        handle_convert("USD:BTC:1000", mock_bot_with_account_manager, mock_parser)

        # Verify commit=True is passed
        call_args = mock_bot_with_account_manager.account_manager.convert.call_args
        assert call_args[1]["commit"] is True

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_shutdown_called_on_success(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser
    ):
        """Test that shutdown is called even on success."""
        handle_convert("USD:BTC:1000", mock_bot_with_account_manager, mock_parser)

        mock_shutdown.assert_called_once()

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_shutdown_called_on_parse_error(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser
    ):
        """Test that shutdown is not called on parse error (early exit)."""
        mock_parser.error.side_effect = SystemExit(2)

        with pytest.raises(SystemExit):
            handle_convert("INVALID", mock_bot_with_account_manager, mock_parser)

        # Shutdown not called because parser.error exits before try block
        mock_shutdown.assert_not_called()

    @patch("bot_v2.cli.commands.convert.ensure_shutdown")
    def test_conversion_with_colons_in_amount(
        self, mock_shutdown, mock_bot_with_account_manager, mock_parser
    ):
        """Test conversion where amount might contain colons (uses maxsplit)."""
        # The split(":", 2) should handle this correctly
        result = handle_convert("USD:BTC:1000:extra", mock_bot_with_account_manager, mock_parser)

        assert result == 0
        # Amount should be "1000:extra" due to maxsplit=2
        mock_bot_with_account_manager.account_manager.convert.assert_called_once_with(
            {"from": "USD", "to": "BTC", "amount": "1000:extra"}, commit=True
        )
