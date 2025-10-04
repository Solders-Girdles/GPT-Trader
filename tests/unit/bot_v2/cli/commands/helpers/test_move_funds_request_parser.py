"""Tests for move funds request parser."""

import argparse
from unittest.mock import Mock

import pytest

from bot_v2.cli.commands.move_funds_request_parser import MoveFundsRequest, MoveFundsRequestParser


class TestMoveFundsRequestParser:
    """Test MoveFundsRequestParser."""

    def test_parse_valid_format(self):
        """Test parsing valid FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT format."""
        parser = Mock(spec=argparse.ArgumentParser)
        request = MoveFundsRequestParser.parse("port-123:port-789:5000", parser)

        assert isinstance(request, MoveFundsRequest)
        assert request.from_portfolio == "port-123"
        assert request.to_portfolio == "port-789"
        assert request.amount == "5000"

    def test_parse_with_whitespace(self):
        """Test parsing strips whitespace from parts."""
        parser = Mock(spec=argparse.ArgumentParser)
        request = MoveFundsRequestParser.parse(" port-123 : port-789 : 5000 ", parser)

        assert request.from_portfolio == "port-123"
        assert request.to_portfolio == "port-789"
        assert request.amount == "5000"

    def test_parse_with_colons_in_amount(self):
        """Test maxsplit=2 allows colons in amount."""
        parser = Mock(spec=argparse.ArgumentParser)
        request = MoveFundsRequestParser.parse("port-123:port-789:1000:extra", parser)

        assert request.from_portfolio == "port-123"
        assert request.to_portfolio == "port-789"
        assert request.amount == "1000:extra"

    def test_parse_invalid_format_missing_parts(self):
        """Test parser.error() called on invalid format."""
        parser = Mock(spec=argparse.ArgumentParser)
        parser.error.side_effect = SystemExit(2)

        with pytest.raises(SystemExit):
            MoveFundsRequestParser.parse("port-123:port-789", parser)

        parser.error.assert_called_once_with(
            "--move-funds requires format FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT"
        )

    def test_parse_invalid_format_single_value(self):
        """Test parser.error() called when no separators."""
        parser = Mock(spec=argparse.ArgumentParser)
        parser.error.side_effect = SystemExit(2)

        with pytest.raises(SystemExit):
            MoveFundsRequestParser.parse("INVALID", parser)

        parser.error.assert_called_once()

    def test_parse_decimal_amount(self):
        """Test parsing decimal amount."""
        parser = Mock(spec=argparse.ArgumentParser)
        request = MoveFundsRequestParser.parse("port-aaa:port-bbb:2500.75", parser)

        assert request.from_portfolio == "port-aaa"
        assert request.to_portfolio == "port-bbb"
        assert request.amount == "2500.75"

    def test_parse_uuid_portfolios(self):
        """Test parsing with UUID-formatted portfolios."""
        parser = Mock(spec=argparse.ArgumentParser)
        uuid1 = "550e8400-e29b-41d4-a716-446655440000"
        uuid2 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        request = MoveFundsRequestParser.parse(f"{uuid1}:{uuid2}:1000", parser)

        assert request.from_portfolio == uuid1
        assert request.to_portfolio == uuid2
        assert request.amount == "1000"


class TestMoveFundsRequest:
    """Test MoveFundsRequest dataclass."""

    def test_move_funds_request_frozen(self):
        """Test MoveFundsRequest is immutable."""
        request = MoveFundsRequest(
            from_portfolio="port-123", to_portfolio="port-789", amount="5000"
        )

        with pytest.raises(AttributeError):
            request.from_portfolio = "port-999"

    def test_move_funds_request_fields(self):
        """Test MoveFundsRequest fields."""
        request = MoveFundsRequest(
            from_portfolio="port-123", to_portfolio="port-789", amount="5000"
        )

        assert request.from_portfolio == "port-123"
        assert request.to_portfolio == "port-789"
        assert request.amount == "5000"
