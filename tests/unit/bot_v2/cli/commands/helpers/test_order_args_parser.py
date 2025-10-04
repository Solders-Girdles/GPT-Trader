"""Unit tests for OrderArgumentsParser."""

import argparse
from argparse import ArgumentParser, Namespace
from decimal import Decimal

import pytest

from bot_v2.cli.commands.order_args import (
    ApplyEditArgs,
    EditPreviewArgs,
    OrderArgumentsParser,
    PreviewOrderArgs,
)
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce


@pytest.fixture
def parser() -> ArgumentParser:
    return argparse.ArgumentParser()


@pytest.fixture
def base_namespace() -> Namespace:
    ns = Namespace()
    ns.order_symbol = "BTC-USD"
    ns.order_side = "buy"
    ns.order_type = "limit"
    ns.order_quantity = 0.5
    ns.order_price = 25_000
    ns.order_stop = None
    ns.order_tif = "gtc"
    ns.order_reduce_only = False
    ns.order_leverage = "2"
    ns.order_client_id = "client-123"
    return ns


def test_parse_preview_success(base_namespace: Namespace, parser: ArgumentParser) -> None:
    result = OrderArgumentsParser.parse_preview(base_namespace, parser)
    assert isinstance(result, PreviewOrderArgs)
    assert result.symbol == "BTC-USD"
    assert result.side is OrderSide.BUY
    assert result.order_type is OrderType.LIMIT
    assert result.quantity == Decimal("0.5")
    assert result.tif is TimeInForce.GTC
    assert result.price == Decimal("25000")
    assert result.stop_price is None
    assert result.leverage == 2
    assert result.client_id == "client-123"


def test_parse_preview_missing_side(base_namespace: Namespace, parser: ArgumentParser) -> None:
    base_namespace.order_side = None
    with pytest.raises(SystemExit):
        OrderArgumentsParser.parse_preview(base_namespace, parser)


def test_parse_preview_invalid_type(base_namespace: Namespace, parser: ArgumentParser) -> None:
    base_namespace.order_type = "invalid"
    with pytest.raises(SystemExit):
        OrderArgumentsParser.parse_preview(base_namespace, parser)


def test_parse_preview_quantity_required(base_namespace: Namespace, parser: ArgumentParser) -> None:
    base_namespace.order_quantity = None
    with pytest.raises(SystemExit):
        OrderArgumentsParser.parse_preview(base_namespace, parser)


def test_parse_edit_preview_success(base_namespace: Namespace, parser: ArgumentParser) -> None:
    base_namespace.edit_order_preview = "order-123"
    result = OrderArgumentsParser.parse_edit_preview(base_namespace, parser)
    assert isinstance(result, EditPreviewArgs)
    assert result.order_id == "order-123"
    assert result.side is OrderSide.BUY


def test_parse_edit_preview_requires_id(base_namespace: Namespace, parser: ArgumentParser) -> None:
    base_namespace.edit_order_preview = None
    with pytest.raises(SystemExit):
        OrderArgumentsParser.parse_edit_preview(base_namespace, parser)


def test_parse_apply_edit_success(parser: ArgumentParser) -> None:
    ns = Namespace(apply_order_edit="order-1:preview-2")
    result = OrderArgumentsParser.parse_apply_edit(ns, parser)
    assert isinstance(result, ApplyEditArgs)
    assert result.order_id == "order-1"
    assert result.preview_id == "preview-2"


def test_parse_apply_edit_invalid_format(parser: ArgumentParser) -> None:
    ns = Namespace(apply_order_edit="invalid-format")
    with pytest.raises(SystemExit):
        OrderArgumentsParser.parse_apply_edit(ns, parser)


def test_parse_preview_no_symbol(base_namespace: Namespace, parser: ArgumentParser) -> None:
    base_namespace.order_symbol = None
    with pytest.raises(SystemExit):
        OrderArgumentsParser.parse_preview(base_namespace, parser)
