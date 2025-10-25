"""Position synchronization helpers for the execution coordinator."""

from __future__ import annotations

import inspect
from typing import Any


class ExecutionCoordinatorPositionMixin:
    """Expose broker position helpers used by orchestration layers."""

    async def sync_positions(self) -> list[Any]:
        broker = self.context.broker
        if broker is None:
            raise RuntimeError("Broker unavailable; cannot sync positions")

        fetch_positions = getattr(broker, "get_positions", None)
        if fetch_positions is None and hasattr(broker, "list_positions"):
            fetch_positions = getattr(broker, "list_positions")
        if fetch_positions is None:
            raise RuntimeError("Broker does not support position synchronization")

        try:
            result = fetch_positions()
            if inspect.isawaitable(result):
                positions = await result  # type: ignore[arg-type]
            else:
                positions = result
        except Exception as exc:
            payload = {"error": str(exc)}
            self._record_event("position_sync_error", payload)
            self._record_event("system_error", payload)
            self._record_event("fallback_position_state", payload)
            self._record_metric("error_count", 1.0, {"type": "position_sync_error"})
            self._record_broker_error(exc)
            raise RuntimeError("Position sync failed") from exc

        if positions is None:
            positions = []
        runtime_state = self.context.runtime_state
        if runtime_state is not None:
            setattr(runtime_state, "positions_snapshot", positions)
        return list(positions)

    async def get_positions(self) -> list[Any]:
        """Return current positions via the broker."""
        try:
            positions = await self.sync_positions()
        except Exception:
            runtime_state = self.context.runtime_state
            if runtime_state is not None:
                snapshot = getattr(runtime_state, "positions_snapshot", None)
                if snapshot is not None:
                    positions = list(snapshot)
                else:
                    raise
            else:
                raise

        positions_list = list(positions or [])
        for pos in positions_list:
            if not hasattr(pos, "size"):
                quantity = getattr(pos, "quantity", None)
                try:
                    setattr(pos, "size", float(quantity) if quantity is not None else None)
                except Exception:
                    setattr(pos, "size", quantity)
        return positions_list


__all__ = ["ExecutionCoordinatorPositionMixin"]
