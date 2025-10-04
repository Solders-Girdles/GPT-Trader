"""Tests for OrderPreviewService."""

import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bot_v2.cli.commands.order_args import PreviewOrderArgs
from bot_v2.cli.commands.order_preview_service import OrderPreviewService
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce


@pytest.fixture
def preview_args() -> PreviewOrderArgs:
    return PreviewOrderArgs(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("0.5"),
        tif=TimeInForce.GTC,
        price=Decimal("25000"),
        stop_price=None,
        reduce_only=False,
        leverage=2,
        client_id="client-123",
    )


@pytest.fixture
def mock_bot() -> SimpleNamespace:
    broker = MagicMock()
    broker.preview_order.return_value = {"order_id": "preview-123", "status": "ACCEPTED"}
    return SimpleNamespace(broker=broker)


def test_preview_success(mock_bot: SimpleNamespace, preview_args: PreviewOrderArgs, capsys) -> None:
    service = OrderPreviewService()
    result = service.preview(mock_bot, preview_args)

    assert result == 0
    mock_bot.broker.preview_order.assert_called_once()

    payload = mock_bot.broker.preview_order.call_args.kwargs
    assert payload["symbol"] == "BTC-USD"
    assert payload["side"] is OrderSide.BUY
    assert payload["quantity"] == Decimal("0.5")

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["order_id"] == "preview-123"


def test_preview_printer_override(
    mock_bot: SimpleNamespace, preview_args: PreviewOrderArgs
) -> None:
    outputs: list[str] = []
    service = OrderPreviewService(printer=outputs.append)

    service.preview(mock_bot, preview_args)

    assert len(outputs) == 1
    assert json.loads(outputs[0])["status"] == "ACCEPTED"


def test_preview_logs_params(
    mock_bot: SimpleNamespace, preview_args: PreviewOrderArgs, caplog
) -> None:
    service = OrderPreviewService()
    with caplog.at_level("DEBUG"):
        service.preview(mock_bot, preview_args)

    assert "Previewing new order" in caplog.text
    assert "Order preview params" in caplog.text
