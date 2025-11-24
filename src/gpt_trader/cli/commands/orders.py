"""Order tooling commands for the CLI."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from dataclasses import asdict
from decimal import Decimal
from typing import Any, Protocol, cast, runtime_checkable

from gpt_trader.cli import options, services
from gpt_trader.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce

_CONFIG_SKIP_KEYS = {
    "orders_command",
    "order_id",
    "preview_id",
    "symbol",
    "side",
    "type",
    "quantity",
    "price",
    "stop",
    "tif",
    "client_id",
    "leverage",
    "reduce_only",
}


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("orders", help="Order preview and edit tooling")
    options.add_profile_option(parser)
    orders_subparsers = parser.add_subparsers(dest="orders_command", required=True)

    preview = orders_subparsers.add_parser("preview", help="Preview a new order and exit")
    options.add_profile_option(preview)
    options.add_order_arguments(preview)
    preview.set_defaults(handler=_handle_preview)

    edit_preview = orders_subparsers.add_parser(
        "edit-preview", help="Preview edits for an existing order"
    )
    options.add_profile_option(edit_preview)
    edit_preview.add_argument("--order-id", required=True, help="Order identifier to edit")
    options.add_order_arguments(edit_preview)
    edit_preview.set_defaults(handler=_handle_edit_preview)

    apply_edit = orders_subparsers.add_parser(
        "apply-edit", help="Apply a previously previewed order edit"
    )
    options.add_profile_option(apply_edit)
    apply_edit.add_argument("--order-id", required=True, help="Order identifier")
    apply_edit.add_argument("--preview-id", required=True, help="Preview identifier to apply")
    apply_edit.set_defaults(handler=_handle_apply_edit)


def _handle_preview(args: Namespace) -> int:
    config = services.build_config_from_args(args, skip=_CONFIG_SKIP_KEYS)
    bot = services.instantiate_bot(config)
    broker = bot.broker
    if not isinstance(broker, OrderPreviewBroker):
        raise RuntimeError("Broker does not support order previews")
    preview_broker = cast(OrderPreviewBroker, broker)
    try:
        payload = _build_order_payload(args)
        result = preview_broker.preview_order(**payload)
        print(json.dumps(result, indent=2, default=str))
    finally:
        asyncio.run(bot.shutdown())
    return 0


def _handle_edit_preview(args: Namespace) -> int:
    config = services.build_config_from_args(args, skip=_CONFIG_SKIP_KEYS)
    bot = services.instantiate_bot(config)
    broker = bot.broker
    if not isinstance(broker, OrderPreviewBroker):
        raise RuntimeError("Broker does not support order edit previews")
    preview_broker = cast(OrderPreviewBroker, broker)
    try:
        payload = _build_order_payload(args)
        result = preview_broker.edit_order_preview(order_id=args.order_id, **payload)
        print(json.dumps(result, indent=2, default=str))
    finally:
        asyncio.run(bot.shutdown())
    return 0


def _handle_apply_edit(args: Namespace) -> int:
    config = services.build_config_from_args(args, skip=_CONFIG_SKIP_KEYS)
    bot = services.instantiate_bot(config)
    broker = bot.broker
    if not isinstance(broker, OrderPreviewBroker):
        raise RuntimeError("Broker does not support order edit application")
    preview_broker = cast(OrderPreviewBroker, broker)
    try:
        order = preview_broker.edit_order(args.order_id, args.preview_id)
        data = asdict(order) if hasattr(order, "__dataclass_fields__") else order
        print(json.dumps(data, indent=2, default=str))
    finally:
        asyncio.run(bot.shutdown())
    return 0


def _build_order_payload(args: Namespace) -> dict[str, object]:
    side = OrderSide[args.side.upper()]
    order_type = OrderType[args.type.upper()]
    quantity = Decimal(str(args.quantity))
    tif = TimeInForce[args.tif.upper()] if args.tif else TimeInForce.GTC

    price = Decimal(str(args.price)) if args.price is not None else None
    stop = Decimal(str(args.stop)) if args.stop is not None else None

    payload: dict[str, object] = {
        "symbol": args.symbol,
        "side": side,
        "order_type": order_type,
        "quantity": quantity,
        "tif": tif,
        "reduce_only": bool(args.reduce_only),
        "leverage": args.leverage,
        "client_id": args.client_id,
    }

    if price is not None:
        payload["price"] = price
    if stop is not None:
        payload["stop_price"] = stop

    return payload


@runtime_checkable
class OrderPreviewBroker(Protocol):
    def preview_order(self, **kwargs: Any) -> Any: ...

    def edit_order_preview(self, order_id: str, **kwargs: Any) -> Any: ...

    def edit_order(self, order_id: str, preview_id: str, **kwargs: Any) -> Any: ...
