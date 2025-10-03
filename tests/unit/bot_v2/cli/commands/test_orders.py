"""Tests for order CLI commands."""

import argparse
import json
from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import MagicMock, Mock, patch

import pytest

from bot_v2.cli.commands.orders import (
    _handle_apply_order_edit,
    _handle_edit_order_preview,
    _handle_preview_order,
    handle_order_tooling,
)
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce


@dataclass
class MockOrder:
    """Mock order for testing."""

    order_id: str
    symbol: str
    side: str
    status: str


@pytest.fixture
def mock_bot():
    """Create a mock PerpsBot instance."""
    bot = Mock()
    bot.broker = Mock()
    bot.broker.preview_order.return_value = {
        "order_id": "preview-123",
        "symbol": "BTC-USD",
        "side": "BUY",
        "quantity": "0.1",
        "price": "50000",
    }
    bot.broker.edit_order_preview.return_value = {
        "preview_id": "edit-preview-456",
        "order_id": "order-123",
        "changes": {"quantity": "0.2"},
    }
    bot.broker.edit_order.return_value = MockOrder(
        order_id="order-123", symbol="BTC-USD", side="BUY", status="OPEN"
    )
    return bot


@pytest.fixture
def parser():
    """Create an ArgumentParser instance."""
    parser = argparse.ArgumentParser()
    return parser


@pytest.fixture
def preview_order_args():
    """Create args for order preview."""
    args = argparse.Namespace()
    args.preview_order = True
    args.edit_order_preview = None
    args.apply_order_edit = None
    args.order_symbol = "BTC-USD"
    args.order_side = "buy"
    args.order_type = "limit"
    args.order_quantity = 0.1
    args.order_price = 50000
    args.order_stop = None
    args.order_tif = "gtc"
    args.order_reduce_only = False
    args.order_leverage = None
    args.order_client_id = None
    return args


@pytest.fixture
def edit_order_preview_args():
    """Create args for order edit preview."""
    args = argparse.Namespace()
    args.preview_order = False
    args.edit_order_preview = "order-123"
    args.apply_order_edit = None
    args.order_symbol = "BTC-USD"
    args.order_side = "buy"
    args.order_type = "limit"
    args.order_quantity = 0.2
    args.order_price = 51000
    args.order_stop = None
    args.order_tif = "gtc"
    args.order_reduce_only = False
    args.order_client_id = None
    return args


@pytest.fixture
def apply_order_edit_args():
    """Create args for applying order edit."""
    args = argparse.Namespace()
    args.preview_order = False
    args.edit_order_preview = None
    args.apply_order_edit = "order-123:preview-456"
    args.order_symbol = "BTC-USD"
    return args


class TestHandleOrderTooling:
    """Tests for handle_order_tooling function."""

    @patch("bot_v2.cli.commands.orders.ensure_shutdown")
    def test_preview_order_flow(self, mock_shutdown, mock_bot, parser, preview_order_args):
        """Test order preview flow."""
        result = handle_order_tooling(preview_order_args, mock_bot, parser)

        assert result == 0
        mock_bot.broker.preview_order.assert_called_once()
        mock_shutdown.assert_called_once_with(mock_bot)

    @patch("bot_v2.cli.commands.orders.ensure_shutdown")
    def test_edit_order_preview_flow(
        self, mock_shutdown, mock_bot, parser, edit_order_preview_args
    ):
        """Test order edit preview flow."""
        result = handle_order_tooling(edit_order_preview_args, mock_bot, parser)

        assert result == 0
        mock_bot.broker.edit_order_preview.assert_called_once()
        mock_shutdown.assert_called_once_with(mock_bot)

    @patch("bot_v2.cli.commands.orders.ensure_shutdown")
    def test_apply_order_edit_flow(self, mock_shutdown, mock_bot, parser, apply_order_edit_args):
        """Test apply order edit flow."""
        result = handle_order_tooling(apply_order_edit_args, mock_bot, parser)

        assert result == 0
        mock_bot.broker.edit_order.assert_called_once()
        mock_shutdown.assert_called_once_with(mock_bot)

    @patch("bot_v2.cli.commands.orders.ensure_shutdown")
    def test_missing_symbol_for_preview(self, mock_shutdown, mock_bot, parser, preview_order_args):
        """Test error when symbol missing for preview."""
        preview_order_args.order_symbol = None

        with pytest.raises(SystemExit):
            handle_order_tooling(preview_order_args, mock_bot, parser)

        # parser.error() exits before finally block, so shutdown is not called

    @patch("bot_v2.cli.commands.orders.ensure_shutdown")
    def test_missing_symbol_for_edit_preview(
        self, mock_shutdown, mock_bot, parser, edit_order_preview_args
    ):
        """Test error when symbol missing for edit preview."""
        edit_order_preview_args.order_symbol = None

        with pytest.raises(SystemExit):
            handle_order_tooling(edit_order_preview_args, mock_bot, parser)

        # parser.error() exits before finally block, so shutdown is not called

    @patch("bot_v2.cli.commands.orders.ensure_shutdown")
    def test_no_action_provided(self, mock_shutdown, mock_bot, parser):
        """Test error when no action is provided."""
        args = argparse.Namespace()
        args.preview_order = False
        args.edit_order_preview = None
        args.apply_order_edit = None
        args.order_symbol = "BTC-USD"

        with pytest.raises(SystemExit):
            handle_order_tooling(args, mock_bot, parser)

        mock_shutdown.assert_called_once_with(mock_bot)

    @patch("bot_v2.cli.commands.orders.ensure_shutdown")
    def test_exception_handling(self, mock_shutdown, mock_bot, parser, preview_order_args):
        """Test exception handling."""
        mock_bot.broker.preview_order.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            handle_order_tooling(preview_order_args, mock_bot, parser)

        mock_shutdown.assert_called_once_with(mock_bot)


class TestHandlePreviewOrder:
    """Tests for _handle_preview_order function."""

    def test_preview_order_success(self, mock_bot, parser, preview_order_args, capsys):
        """Test successful order preview."""
        result = _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        mock_bot.broker.preview_order.assert_called_once()

        # Check that preview was called with correct parameters
        call_kwargs = mock_bot.broker.preview_order.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC-USD"
        assert call_kwargs["side"] == OrderSide.BUY
        assert call_kwargs["order_type"] == OrderType.LIMIT
        assert call_kwargs["quantity"] == Decimal("0.1")
        assert call_kwargs["price"] == Decimal("50000")
        assert call_kwargs["tif"] == TimeInForce.GTC

        # Check output
        captured = capsys.readouterr()
        assert "preview-123" in captured.out

    def test_preview_order_market_order(self, mock_bot, parser, preview_order_args, capsys):
        """Test market order preview (no price)."""
        preview_order_args.order_type = "market"
        preview_order_args.order_price = None

        result = _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.preview_order.call_args.kwargs
        assert call_kwargs["order_type"] == OrderType.MARKET
        assert call_kwargs["price"] is None

    def test_preview_order_with_stop_price(self, mock_bot, parser, preview_order_args):
        """Test order preview with stop price."""
        preview_order_args.order_stop = 49000

        result = _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.preview_order.call_args.kwargs
        assert call_kwargs["stop_price"] == Decimal("49000")

    def test_preview_order_with_custom_tif(self, mock_bot, parser, preview_order_args):
        """Test order preview with custom time in force."""
        preview_order_args.order_tif = "ioc"

        result = _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.preview_order.call_args.kwargs
        assert call_kwargs["tif"] == TimeInForce.IOC

    def test_preview_order_reduce_only(self, mock_bot, parser, preview_order_args):
        """Test reduce-only order preview."""
        preview_order_args.order_reduce_only = True

        result = _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.preview_order.call_args.kwargs
        assert call_kwargs["reduce_only"] is True

    def test_preview_order_with_leverage(self, mock_bot, parser, preview_order_args):
        """Test order preview with leverage."""
        preview_order_args.order_leverage = 10

        result = _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.preview_order.call_args.kwargs
        assert call_kwargs["leverage"] == 10

    def test_preview_order_with_client_id(self, mock_bot, parser, preview_order_args):
        """Test order preview with client ID."""
        preview_order_args.order_client_id = "my-client-123"

        result = _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.preview_order.call_args.kwargs
        assert call_kwargs["client_id"] == "my-client-123"

    def test_preview_order_sell_side(self, mock_bot, parser, preview_order_args):
        """Test sell order preview."""
        preview_order_args.order_side = "sell"

        result = _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.preview_order.call_args.kwargs
        assert call_kwargs["side"] == OrderSide.SELL

    def test_preview_order_missing_side(self, mock_bot, parser, preview_order_args):
        """Test error when side is missing."""
        preview_order_args.order_side = None

        with pytest.raises(SystemExit):
            _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

    def test_preview_order_missing_type(self, mock_bot, parser, preview_order_args):
        """Test error when order type is missing."""
        preview_order_args.order_type = None

        with pytest.raises(SystemExit):
            _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

    def test_preview_order_missing_quantity(self, mock_bot, parser, preview_order_args):
        """Test error when quantity is missing."""
        preview_order_args.order_quantity = None

        with pytest.raises(SystemExit):
            _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")


class TestHandleEditOrderPreview:
    """Tests for _handle_edit_order_preview function."""

    def test_edit_order_preview_success(self, mock_bot, parser, edit_order_preview_args, capsys):
        """Test successful order edit preview."""
        result = _handle_edit_order_preview(edit_order_preview_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        mock_bot.broker.edit_order_preview.assert_called_once()

        # Check that preview was called with correct parameters
        call_kwargs = mock_bot.broker.edit_order_preview.call_args.kwargs
        assert call_kwargs["order_id"] == "order-123"
        assert call_kwargs["symbol"] == "BTC-USD"
        assert call_kwargs["side"] == OrderSide.BUY
        assert call_kwargs["order_type"] == OrderType.LIMIT
        assert call_kwargs["quantity"] == Decimal("0.2")
        assert call_kwargs["price"] == Decimal("51000")

        # Check output
        captured = capsys.readouterr()
        assert "edit-preview-456" in captured.out

    def test_edit_order_preview_with_stop(self, mock_bot, parser, edit_order_preview_args):
        """Test edit preview with stop price."""
        edit_order_preview_args.order_stop = 50000

        result = _handle_edit_order_preview(edit_order_preview_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.edit_order_preview.call_args.kwargs
        assert call_kwargs["stop_price"] == Decimal("50000")

    def test_edit_order_preview_with_client_id(self, mock_bot, parser, edit_order_preview_args):
        """Test edit preview with new client ID."""
        edit_order_preview_args.order_client_id = "new-client-456"

        result = _handle_edit_order_preview(edit_order_preview_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.edit_order_preview.call_args.kwargs
        assert call_kwargs["new_client_id"] == "new-client-456"

    def test_edit_order_preview_reduce_only(self, mock_bot, parser, edit_order_preview_args):
        """Test edit preview with reduce only flag."""
        edit_order_preview_args.order_reduce_only = True

        result = _handle_edit_order_preview(edit_order_preview_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.edit_order_preview.call_args.kwargs
        assert call_kwargs["reduce_only"] is True

    def test_edit_order_preview_missing_side(self, mock_bot, parser, edit_order_preview_args):
        """Test error when side is missing."""
        edit_order_preview_args.order_side = None

        with pytest.raises(SystemExit):
            _handle_edit_order_preview(edit_order_preview_args, mock_bot, parser, "BTC-USD")

    def test_edit_order_preview_missing_type(self, mock_bot, parser, edit_order_preview_args):
        """Test error when order type is missing."""
        edit_order_preview_args.order_type = None

        with pytest.raises(SystemExit):
            _handle_edit_order_preview(edit_order_preview_args, mock_bot, parser, "BTC-USD")

    def test_edit_order_preview_missing_quantity(self, mock_bot, parser, edit_order_preview_args):
        """Test error when quantity is missing."""
        edit_order_preview_args.order_quantity = None

        with pytest.raises(SystemExit):
            _handle_edit_order_preview(edit_order_preview_args, mock_bot, parser, "BTC-USD")


class TestHandleApplyOrderEdit:
    """Tests for _handle_apply_order_edit function."""

    def test_apply_order_edit_success(self, mock_bot, parser, apply_order_edit_args, capsys):
        """Test successful order edit application."""
        result = _handle_apply_order_edit(apply_order_edit_args, mock_bot, parser)

        assert result == 0
        mock_bot.broker.edit_order.assert_called_once_with("order-123", "preview-456")

        # Check output
        captured = capsys.readouterr()
        assert "order-123" in captured.out

    def test_apply_order_edit_with_whitespace(self, mock_bot, parser, apply_order_edit_args):
        """Test order edit with whitespace in argument."""
        apply_order_edit_args.apply_order_edit = " order-123 : preview-456 "

        result = _handle_apply_order_edit(apply_order_edit_args, mock_bot, parser)

        assert result == 0
        mock_bot.broker.edit_order.assert_called_once_with("order-123", "preview-456")

    def test_apply_order_edit_invalid_format_no_colon(
        self, mock_bot, parser, apply_order_edit_args
    ):
        """Test error when format is invalid (no colon)."""
        apply_order_edit_args.apply_order_edit = "order-123-preview-456"

        with pytest.raises(SystemExit):
            _handle_apply_order_edit(apply_order_edit_args, mock_bot, parser)

    def test_apply_order_edit_with_empty_parts(self, mock_bot, parser, apply_order_edit_args):
        """Test with empty parts (still valid split but empty strings)."""
        apply_order_edit_args.apply_order_edit = ":"

        # This actually succeeds with empty strings
        result = _handle_apply_order_edit(apply_order_edit_args, mock_bot, parser)

        assert result == 0
        mock_bot.broker.edit_order.assert_called_once_with("", "")


class TestOrderParameters:
    """Tests for order parameter parsing."""

    def test_decimal_conversion(self, mock_bot, parser, preview_order_args):
        """Test that numeric values are converted to Decimal."""
        preview_order_args.order_quantity = "0.123456789"
        preview_order_args.order_price = "50000.50"

        result = _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.preview_order.call_args.kwargs
        assert isinstance(call_kwargs["quantity"], Decimal)
        assert isinstance(call_kwargs["price"], Decimal)
        assert call_kwargs["quantity"] == Decimal("0.123456789")
        assert call_kwargs["price"] == Decimal("50000.50")

    def test_case_insensitive_enums(self, mock_bot, parser, preview_order_args):
        """Test that enum values are case-insensitive."""
        preview_order_args.order_side = "BUY"
        preview_order_args.order_type = "LIMIT"
        preview_order_args.order_tif = "GTC"

        result = _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        assert result == 0
        call_kwargs = mock_bot.broker.preview_order.call_args.kwargs
        assert call_kwargs["side"] == OrderSide.BUY
        assert call_kwargs["order_type"] == OrderType.LIMIT
        assert call_kwargs["tif"] == TimeInForce.GTC


class TestOutputFormatting:
    """Tests for output formatting."""

    def test_preview_output_is_json(self, mock_bot, parser, preview_order_args, capsys):
        """Test that preview output is valid JSON."""
        _handle_preview_order(preview_order_args, mock_bot, parser, "BTC-USD")

        captured = capsys.readouterr()
        # Should be able to parse as JSON
        parsed = json.loads(captured.out)
        assert "order_id" in parsed

    def test_edit_preview_output_is_json(self, mock_bot, parser, edit_order_preview_args, capsys):
        """Test that edit preview output is valid JSON."""
        _handle_edit_order_preview(edit_order_preview_args, mock_bot, parser, "BTC-USD")

        captured = capsys.readouterr()
        # Should be able to parse as JSON
        parsed = json.loads(captured.out)
        assert "preview_id" in parsed

    def test_apply_edit_output_is_json(self, mock_bot, parser, apply_order_edit_args, capsys):
        """Test that apply edit output is valid JSON."""
        _handle_apply_order_edit(apply_order_edit_args, mock_bot, parser)

        captured = capsys.readouterr()
        # Should be able to parse as JSON
        parsed = json.loads(captured.out)
        assert "order_id" in parsed
