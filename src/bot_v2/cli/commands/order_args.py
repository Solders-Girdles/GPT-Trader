"""Argument sanitizers for order CLI commands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce


@dataclass(frozen=True)
class PreviewOrderArgs:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    tif: TimeInForce
    price: Decimal | None
    stop_price: Decimal | None
    reduce_only: bool
    leverage: int | None
    client_id: str | None


@dataclass(frozen=True)
class EditPreviewArgs(PreviewOrderArgs):
    order_id: str


@dataclass(frozen=True)
class ApplyEditArgs:
    order_id: str
    preview_id: str


class OrderArgumentsParser:
    """Parses and validates CLI arguments for order commands."""

    @staticmethod
    def _require(parser: argparse.ArgumentParser, condition: bool, message: str) -> None:
        if not condition:
            parser.error(message)

    @staticmethod
    def _parse_symbol(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
        symbol = getattr(args, "order_symbol", None)
        OrderArgumentsParser._require(parser, bool(symbol), "--order-symbol is required")
        return symbol.strip()

    @staticmethod
    def _parse_side(value: str | None, parser: argparse.ArgumentParser) -> OrderSide:
        OrderArgumentsParser._require(parser, value is not None, "--order-side is required")
        try:
            return OrderSide[value.upper()]
        except KeyError as exc:
            parser.error(f"Invalid order side: {value}")
            raise exc

    @staticmethod
    def _parse_type(value: str | None, parser: argparse.ArgumentParser) -> OrderType:
        OrderArgumentsParser._require(parser, value is not None, "--order-type is required")
        try:
            return OrderType[value.upper()]
        except KeyError as exc:
            parser.error(f"Invalid order type: {value}")
            raise exc

    @staticmethod
    def _parse_tif(value: str | None) -> TimeInForce:
        if value is None:
            return TimeInForce.GTC
        try:
            return TimeInForce[value.upper()]
        except KeyError:
            return TimeInForce.GTC

    @staticmethod
    def _parse_decimal(value: float | int | str | None) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value))

    @staticmethod
    def _parse_int(value: float | int | str | None) -> int | None:
        if value is None:
            return None
        return int(value)

    @classmethod
    def parse_preview(
        cls, args: argparse.Namespace, parser: argparse.ArgumentParser
    ) -> PreviewOrderArgs:
        symbol = cls._parse_symbol(args, parser)
        side = cls._parse_side(getattr(args, "order_side", None), parser)
        order_type = cls._parse_type(getattr(args, "order_type", None), parser)
        quantity = cls._parse_decimal(getattr(args, "order_quantity", None))
        cls._require(parser, quantity is not None, "--order-quantity is required")

        tif = cls._parse_tif(getattr(args, "order_tif", None))
        price = cls._parse_decimal(getattr(args, "order_price", None))
        stop_price = cls._parse_decimal(getattr(args, "order_stop", None))
        leverage = cls._parse_int(getattr(args, "order_leverage", None))

        return PreviewOrderArgs(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            tif=tif,
            price=price,
            stop_price=stop_price,
            reduce_only=bool(getattr(args, "order_reduce_only", False)),
            leverage=leverage,
            client_id=getattr(args, "order_client_id", None),
        )

    @classmethod
    def parse_edit_preview(
        cls, args: argparse.Namespace, parser: argparse.ArgumentParser
    ) -> EditPreviewArgs:
        preview_args = cls.parse_preview(args, parser)
        order_id = getattr(args, "edit_order_preview", None)
        cls._require(parser, bool(order_id), "--edit-order-preview requires ORDER_ID")
        return EditPreviewArgs(order_id=order_id.strip(), **preview_args.__dict__)

    @classmethod
    def parse_apply_edit(
        cls, args: argparse.Namespace, parser: argparse.ArgumentParser
    ) -> ApplyEditArgs:
        raw_value = getattr(args, "apply_order_edit", None)
        cls._require(parser, bool(raw_value), "--apply-order-edit requires ORDER_ID:PREVIEW_ID")
        try:
            order_id, preview_id = (part.strip() for part in raw_value.split(":", 1))
        except ValueError as exc:
            parser.error("--apply-order-edit requires ORDER_ID:PREVIEW_ID")
            raise exc
        return ApplyEditArgs(order_id=order_id, preview_id=preview_id)
