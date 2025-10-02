"""Tests for move-funds CLI command."""

import argparse
import json
from unittest.mock import Mock, patch

import pytest

from bot_v2.cli.commands.move_funds import handle_move_funds


@pytest.fixture
def mock_bot_with_account_manager():
    """Create a mock PerpsBot with account manager."""
    bot = Mock()
    account_mgr = Mock()
    account_mgr.move_funds.return_value = {
        "transfer_id": "xfer-456",
        "from_portfolio": "port-123",
        "to_portfolio": "port-789",
        "amount": "5000.00",
        "status": "completed"
    }
    bot.account_manager = account_mgr
    return bot


@pytest.fixture
def mock_parser():
    """Create a mock ArgumentParser."""
    parser = Mock(spec=argparse.ArgumentParser)
    return parser


class TestHandleMoveFunds:
    """Tests for handle_move_funds function."""

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_successful_fund_movement(self, mock_shutdown, mock_bot_with_account_manager, mock_parser, capsys):
        """Test successful fund movement between portfolios."""
        result = handle_move_funds("port-123:port-789:5000", mock_bot_with_account_manager, mock_parser)

        assert result == 0
        mock_bot_with_account_manager.account_manager.move_funds.assert_called_once_with(
            {"from_portfolio": "port-123", "to_portfolio": "port-789", "amount": "5000"}
        )
        mock_shutdown.assert_called_once_with(mock_bot_with_account_manager)

        # Check output
        captured = capsys.readouterr()
        assert "xfer-456" in captured.out
        assert "port-123" in captured.out
        assert "port-789" in captured.out

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_move_funds_output_is_json(self, mock_shutdown, mock_bot_with_account_manager, mock_parser, capsys):
        """Test that move-funds output is valid JSON."""
        handle_move_funds("port-123:port-789:5000", mock_bot_with_account_manager, mock_parser)

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "transfer_id" in parsed
        assert parsed["from_portfolio"] == "port-123"
        assert parsed["to_portfolio"] == "port-789"

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_invalid_format_missing_parts(self, mock_shutdown, mock_bot_with_account_manager, mock_parser, caplog):
        """Test error when move-funds format is invalid (missing parts)."""
        import logging

        mock_parser.error.side_effect = SystemExit(2)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                handle_move_funds("port-123:port-789", mock_bot_with_account_manager, mock_parser)

        # Verify error message is logged
        assert "Invalid move-funds argument format" in caplog.text
        assert "port-123:port-789" in caplog.text
        mock_parser.error.assert_called_once_with(
            "--move-funds requires format FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT"
        )

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_invalid_format_single_value(self, mock_shutdown, mock_bot_with_account_manager, mock_parser):
        """Test error when move-funds format has no separators."""
        mock_parser.error.side_effect = SystemExit(2)

        with pytest.raises(SystemExit):
            handle_move_funds("INVALID", mock_bot_with_account_manager, mock_parser)

        mock_parser.error.assert_called_once()

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_move_funds_with_whitespace(self, mock_shutdown, mock_bot_with_account_manager, mock_parser):
        """Test fund movement with whitespace around values."""
        result = handle_move_funds(" port-123 : port-789 : 5000 ", mock_bot_with_account_manager, mock_parser)

        assert result == 0
        mock_bot_with_account_manager.account_manager.move_funds.assert_called_once_with(
            {"from_portfolio": "port-123", "to_portfolio": "port-789", "amount": "5000"}
        )

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_move_funds_exception_raised(self, mock_shutdown, mock_bot_with_account_manager, mock_parser, caplog):
        """Test exception during fund movement is raised and logged."""
        import logging

        mock_bot_with_account_manager.account_manager.move_funds.side_effect = Exception("Insufficient balance")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception, match="Insufficient balance"):
                handle_move_funds("port-123:port-789:5000", mock_bot_with_account_manager, mock_parser)

        # Verify error was logged
        assert "Fund movement failed" in caplog.text
        assert "Insufficient balance" in caplog.text
        # Shutdown should still be called in finally block
        mock_shutdown.assert_called_once_with(mock_bot_with_account_manager)

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_move_funds_with_decimal_amount(self, mock_shutdown, mock_bot_with_account_manager, mock_parser):
        """Test fund movement with decimal amount."""
        mock_bot_with_account_manager.account_manager.move_funds.return_value = {
            "from_portfolio": "port-aaa",
            "to_portfolio": "port-bbb",
            "amount": "2500.75",
            "status": "completed"
        }

        result = handle_move_funds("port-aaa:port-bbb:2500.75", mock_bot_with_account_manager, mock_parser)

        assert result == 0
        mock_bot_with_account_manager.account_manager.move_funds.assert_called_once_with(
            {"from_portfolio": "port-aaa", "to_portfolio": "port-bbb", "amount": "2500.75"}
        )

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_shutdown_called_on_success(self, mock_shutdown, mock_bot_with_account_manager, mock_parser):
        """Test that shutdown is called even on success."""
        handle_move_funds("port-123:port-789:5000", mock_bot_with_account_manager, mock_parser)

        mock_shutdown.assert_called_once()

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_shutdown_called_on_error(self, mock_shutdown, mock_bot_with_account_manager, mock_parser):
        """Test that shutdown is called even when error occurs."""
        mock_bot_with_account_manager.account_manager.move_funds.side_effect = Exception("API error")

        with pytest.raises(Exception):
            handle_move_funds("port-123:port-789:5000", mock_bot_with_account_manager, mock_parser)

        mock_shutdown.assert_called_once()

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_move_funds_with_uuid_portfolios(self, mock_shutdown, mock_bot_with_account_manager, mock_parser):
        """Test fund movement with UUID-formatted portfolios."""
        uuid1 = "550e8400-e29b-41d4-a716-446655440000"
        uuid2 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

        result = handle_move_funds(f"{uuid1}:{uuid2}:1000", mock_bot_with_account_manager, mock_parser)

        assert result == 0
        mock_bot_with_account_manager.account_manager.move_funds.assert_called_once_with(
            {"from_portfolio": uuid1, "to_portfolio": uuid2, "amount": "1000"}
        )

    @patch("bot_v2.cli.commands.move_funds.ensure_shutdown")
    def test_move_funds_with_colons_in_amount(self, mock_shutdown, mock_bot_with_account_manager, mock_parser):
        """Test fund movement where amount might contain colons (uses maxsplit)."""
        result = handle_move_funds("port-123:port-789:1000:extra", mock_bot_with_account_manager, mock_parser)

        assert result == 0
        # Amount should be "1000:extra" due to maxsplit=2
        mock_bot_with_account_manager.account_manager.move_funds.assert_called_once_with(
            {"from_portfolio": "port-123", "to_portfolio": "port-789", "amount": "1000:extra"}
        )
