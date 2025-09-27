#!/usr/bin/env python3
"""CLI helper to preview or edit orders without launching the full bot."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal

from bot_v2.orchestration.perps_bot import BotConfig, PerpsBot
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce


def positive_decimal(value: str) -> Decimal:
    v = Decimal(value)
    if v <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return v


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Order preview/edit helper")
    parser.add_argument("--profile", default="dev", choices=["dev", "demo", "canary", "prod"], help="Bot profile to load for credentials")
    parser.add_argument("--symbol", required=True, help="Product symbol, e.g. BTC-PERP")
    parser.add_argument("--side", choices=[s.value for s in OrderSide], required=True, help="Order side (buy/sell)")
    parser.add_argument("--type", dest="order_type", choices=[t.value for t in OrderType if t in (OrderType.MARKET, OrderType.LIMIT, OrderType.STOP_LIMIT)], required=True, help="Order type")
    parser.add_argument("--qty", type=positive_decimal, required=True, help="Order size in base units")
    parser.add_argument("--price", type=positive_decimal, help="Limit price (required for limit)" )
    parser.add_argument("--stop", type=positive_decimal, help="Stop price for stop-limit")
    parser.add_argument("--tif", choices=[t.value.upper() for t in TimeInForce if t != TimeInForce.GTC] + ["GTC"], default="GTC", help="Time in force")
    parser.add_argument("--reduce-only", action="store_true", help="Set reduce-only flag")
    parser.add_argument("--leverage", type=int, help="Target leverage override")
    parser.add_argument("--client-id", help="Client order id to reuse")
    parser.add_argument("--preview", action="store_true", help="Preview a new order (default action)")
    parser.add_argument("--edit-preview", metavar="ORDER_ID", help="Preview edits for an existing order id")
    parser.add_argument("--apply-edit", metavar="ORDER_ID:PREVIEW_ID", help="Apply order edit using preview id")
    parser.add_argument("--post-only", action="store_true", help="Set post_only flag for limit orders")
    parser.add_argument("--json", action="store_true", help="Output raw JSON without formatting")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.preview and args.edit_preview:
        parser.error("Specify only one of --preview or --edit-preview")
    if args.apply_edit and not args.apply_edit.count(":"):
        parser.error("--apply-edit requires ORDER_ID:PREVIEW_ID")

    cfg = BotConfig.from_profile(args.profile)
    cfg.dry_run = True
    bot = PerpsBot(cfg)

    try:
        side = OrderSide(args.side)
        order_type = OrderType(args.order_type)
        tif = TimeInForce[args.tif.upper()]
        payload = {
            "symbol": args.symbol,
            "side": side,
            "order_type": order_type,
            "qty": args.qty,
            "price": args.price,
            "stop_price": args.stop,
            "tif": tif,
            "reduce_only": args.reduce_only,
            "leverage": args.leverage,
            "post_only": args.post_only,
        }

        if args.apply_edit:
            order_id, preview_id = [part.strip() for part in args.apply_edit.split(":", 1)]
            result = bot.broker.edit_order(order_id, preview_id)
        elif args.edit_preview:
            payload.update({"order_id": args.edit_preview, "new_client_id": args.client_id})
            result = bot.broker.edit_order_preview(**{k: v for k, v in payload.items() if k != "symbol"}, symbol=args.symbol)
        else:
            result = bot.broker.preview_order(**payload, client_id=args.client_id)

        text = json.dumps(result, indent=None if args.json else 2, default=str)
        print(text)
        return 0
    finally:
        import asyncio
        asyncio.run(bot.shutdown())


if __name__ == "__main__":
    raise SystemExit(main())
