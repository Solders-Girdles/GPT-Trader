"""Tests for convert request parser."""

import argparse
from unittest.mock import Mock

import pytest

from bot_v2.cli.commands.convert_request_parser import ConvertRequest, ConvertRequestParser


class TestConvertRequestParser:
    """Test ConvertRequestParser."""

    def test_parse_valid_format(self):
        """Test parsing valid FROM:TO:AMOUNT format."""
        parser = Mock(spec=argparse.ArgumentParser)
        request = ConvertRequestParser.parse("USD:BTC:1000", parser)

        assert isinstance(request, ConvertRequest)
        assert request.from_asset == "USD"
        assert request.to_asset == "BTC"
        assert request.amount == "1000"

    def test_parse_with_whitespace(self):
        """Test parsing strips whitespace from parts."""
        parser = Mock(spec=argparse.ArgumentParser)
        request = ConvertRequestParser.parse(" USD : BTC : 1000 ", parser)

        assert request.from_asset == "USD"
        assert request.to_asset == "BTC"
        assert request.amount == "1000"

    def test_parse_with_colons_in_amount(self):
        """Test maxsplit=2 allows colons in amount."""
        parser = Mock(spec=argparse.ArgumentParser)
        request = ConvertRequestParser.parse("USD:BTC:1000:extra", parser)

        assert request.from_asset == "USD"
        assert request.to_asset == "BTC"
        assert request.amount == "1000:extra"

    def test_parse_invalid_format_missing_parts(self):
        """Test parser.error() called on invalid format."""
        parser = Mock(spec=argparse.ArgumentParser)
        parser.error.side_effect = SystemExit(2)

        with pytest.raises(SystemExit):
            ConvertRequestParser.parse("USD:BTC", parser)

        parser.error.assert_called_once_with("--convert requires format FROM:TO:AMOUNT")

    def test_parse_invalid_format_single_value(self):
        """Test parser.error() called when no separators."""
        parser = Mock(spec=argparse.ArgumentParser)
        parser.error.side_effect = SystemExit(2)

        with pytest.raises(SystemExit):
            ConvertRequestParser.parse("INVALID", parser)

        parser.error.assert_called_once()

    def test_parse_decimal_amount(self):
        """Test parsing decimal amount."""
        parser = Mock(spec=argparse.ArgumentParser)
        request = ConvertRequestParser.parse("BTC:ETH:0.5", parser)

        assert request.from_asset == "BTC"
        assert request.to_asset == "ETH"
        assert request.amount == "0.5"


class TestConvertRequest:
    """Test ConvertRequest dataclass."""

    def test_convert_request_frozen(self):
        """Test ConvertRequest is immutable."""
        request = ConvertRequest(from_asset="USD", to_asset="BTC", amount="1000")

        with pytest.raises(AttributeError):
            request.from_asset = "EUR"

    def test_convert_request_fields(self):
        """Test ConvertRequest fields."""
        request = ConvertRequest(from_asset="USD", to_asset="BTC", amount="1000")

        assert request.from_asset == "USD"
        assert request.to_asset == "BTC"
        assert request.amount == "1000"
