"""Services for handling order edit preview and apply CLI flows."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from bot_v2.cli.commands.order_args import ApplyEditArgs, EditPreviewArgs


class EditPreviewService:
    """Executes broker edit preview/apply calls and renders output for CLI usage."""

    def __init__(self, printer: Callable[[str], None] | None = None) -> None:
        self._printer = printer or print
        self._logger = logging.getLogger(__name__)

    def edit_preview(self, bot: Any, args: EditPreviewArgs) -> int:
        """Execute edit order preview flow and print JSON response."""
        self._logger.info(
            "Previewing order edit for order_id=%s, symbol=%s", args.order_id, args.symbol
        )
        self._logger.debug(
            "Edit preview params: side=%s, type=%s, qty=%s, price=%s, stop=%s, tif=%s",
            args.side,
            args.order_type,
            args.quantity,
            args.price,
            args.stop_price,
            args.tif,
        )

        # Preview the edit
        preview = bot.broker.edit_order_preview(
            order_id=args.order_id,
            symbol=args.symbol,
            side=args.side,
            order_type=args.order_type,
            quantity=args.quantity,
            price=args.price,
            stop_price=args.stop_price,
            tif=args.tif,
            new_client_id=args.client_id,
            reduce_only=args.reduce_only,
        )

        self._logger.info("Order edit preview completed successfully")

        # Print preview as formatted JSON
        output = json.dumps(preview, indent=2, default=str)
        self._printer(output)

        return 0

    def apply_edit(self, bot: Any, args: ApplyEditArgs) -> int:
        """Execute apply order edit flow and print JSON response."""
        self._logger.info(
            "Applying order edit for order_id=%s, preview_id=%s", args.order_id, args.preview_id
        )

        # Apply the edit
        order = bot.broker.edit_order(args.order_id, args.preview_id)

        self._logger.info("Order edit applied successfully: order_id=%s", order.order_id)

        # Print result as formatted JSON
        output = json.dumps(asdict(order), indent=2, default=str)
        self._printer(output)

        return 0
