"""Order lock management for execution coordinator."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from ..logging_utils import logger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.coordinators.base import CoordinatorContext
    from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState


class OrderLockMixin:
    """Ensure an async lock exists for order placement."""

    context: CoordinatorContext

    def ensure_order_lock(self) -> asyncio.Lock:
        runtime_state_obj = self.context.runtime_state
        if runtime_state_obj is None:
            raise RuntimeError("Runtime state is unavailable; cannot create order lock")

        runtime_state = cast("PerpsBotRuntimeState", runtime_state_obj)

        if runtime_state.order_lock is None:
            try:
                runtime_state.order_lock = asyncio.Lock()
            except RuntimeError as exc:
                logger.error(
                    "Unable to initialize async order lock: %s",
                    exc,
                    operation="order_lock",
                    stage="initialize",
                )
                raise
        return cast(asyncio.Lock, runtime_state.order_lock)


__all__ = ["OrderLockMixin"]
