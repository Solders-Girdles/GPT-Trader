"""Tests for EditPreviewService."""

import json
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bot_v2.cli.commands.edit_preview_service import EditPreviewService
from bot_v2.cli.commands.order_args import ApplyEditArgs, EditPreviewArgs
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce


@pytest.fixture
def edit_preview_args() -> EditPreviewArgs:
    return EditPreviewArgs(
        order_id="order-123",
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("26000"),
        stop_price=None,
        tif=TimeInForce.GTC,
        client_id="client-456",
        reduce_only=False,
        leverage=2,
    )


@pytest.fixture
def apply_edit_args() -> ApplyEditArgs:
    return ApplyEditArgs(
        order_id="order-123",
        preview_id="preview-456",
    )


@pytest.fixture
def mock_bot_with_edit() -> SimpleNamespace:
    broker = MagicMock()
    broker.edit_order_preview.return_value = {
        "preview_id": "preview-456",
        "order_id": "order-123",
        "status": "PENDING",
    }

    @dataclass
    class MockOrder:
        order_id: str
        status: str

    broker.edit_order.return_value = MockOrder(order_id="order-123", status="FILLED")
    return SimpleNamespace(broker=broker)


def test_edit_preview_success(
    mock_bot_with_edit: SimpleNamespace, edit_preview_args: EditPreviewArgs, capsys
) -> None:
    service = EditPreviewService()
    result = service.edit_preview(mock_bot_with_edit, edit_preview_args)

    assert result == 0
    mock_bot_with_edit.broker.edit_order_preview.assert_called_once()

    call_kwargs = mock_bot_with_edit.broker.edit_order_preview.call_args.kwargs
    assert call_kwargs["order_id"] == "order-123"
    assert call_kwargs["symbol"] == "BTC-USD"
    assert call_kwargs["side"] is OrderSide.BUY
    assert call_kwargs["quantity"] == Decimal("1.0")

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["preview_id"] == "preview-456"


def test_edit_preview_printer_override(
    mock_bot_with_edit: SimpleNamespace, edit_preview_args: EditPreviewArgs
) -> None:
    outputs: list[str] = []
    service = EditPreviewService(printer=outputs.append)

    service.edit_preview(mock_bot_with_edit, edit_preview_args)

    assert len(outputs) == 1
    assert json.loads(outputs[0])["status"] == "PENDING"


def test_edit_preview_logs_params(
    mock_bot_with_edit: SimpleNamespace, edit_preview_args: EditPreviewArgs, caplog
) -> None:
    service = EditPreviewService()
    with caplog.at_level("DEBUG"):
        service.edit_preview(mock_bot_with_edit, edit_preview_args)

    assert "Previewing order edit" in caplog.text
    assert "Edit preview params" in caplog.text


def test_apply_edit_success(
    mock_bot_with_edit: SimpleNamespace, apply_edit_args: ApplyEditArgs, capsys
) -> None:
    service = EditPreviewService()
    result = service.apply_edit(mock_bot_with_edit, apply_edit_args)

    assert result == 0
    mock_bot_with_edit.broker.edit_order.assert_called_once_with("order-123", "preview-456")

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["order_id"] == "order-123"
    assert parsed["status"] == "FILLED"


def test_apply_edit_printer_override(
    mock_bot_with_edit: SimpleNamespace, apply_edit_args: ApplyEditArgs
) -> None:
    outputs: list[str] = []
    service = EditPreviewService(printer=outputs.append)

    service.apply_edit(mock_bot_with_edit, apply_edit_args)

    assert len(outputs) == 1
    parsed = json.loads(outputs[0])
    assert parsed["order_id"] == "order-123"
    assert parsed["status"] == "FILLED"
