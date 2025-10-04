"""Services for handling order preview CLI flows."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from bot_v2.cli.commands.order_args import PreviewOrderArgs


class OrderPreviewService:
    """Executes broker preview calls and renders output for CLI usage."""

    def __init__(self, printer: Callable[[str], None] | None = None) -> None:
        self._printer = printer or print
        self._logger = logging.getLogger(__name__)

    def preview(self, bot: Any, args: PreviewOrderArgs) -> int:
        """Execute preview order flow and print JSON response."""
        self._logger.info("Previewing new order for symbol=%s", args.symbol)
        self._logger.debug(
            "Order preview params: side=%s, type=%s, qty=%s, price=%s, stop=%s, tif=%s",
            args.side,
            args.order_type,
            args.quantity,
            args.price,
            args.stop_price,
            args.tif,
        )

        payload = {
            "symbol": args.symbol,
            "side": args.side,
            "order_type": args.order_type,
            "quantity": args.quantity,
            "price": args.price,
            "stop_price": args.stop_price,
            "tif": args.tif,
            "reduce_only": args.reduce_only,
            "leverage": args.leverage,
            "client_id": args.client_id,
        }

        data = bot.broker.preview_order(**payload)
        self._logger.info("Order preview completed successfully")

        output = json.dumps(data, indent=2, default=str)
        self._printer(output)
        return 0
