"""Position reconciliation helpers for SystemMonitor."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.quantities import quantity_from
from gpt_trader.utilities.telemetry import emit_metric

logger = get_logger(__name__, component="system_monitor_positions")


class PositionReconciler:
    """Detects position drift by comparing broker state against cached state."""

    def __init__(self, *, event_store: Any, bot_id: str) -> None:
        self._event_store = event_store
        self._bot_id = bot_id

    async def run(self, bot: Any, interval_seconds: int = 90) -> None:
        """Continuously reconcile positions while the bot is running."""

        while bot.running:
            try:
                positions = await self._fetch_positions(bot)
                current = self._normalize_positions(positions)

                state = bot.runtime_state
                if not state.last_positions and current:
                    state.last_positions = current
                else:
                    changes = self._calculate_diff(state.last_positions, current)
                    if changes:
                        self._emit_position_changes(bot, changes)
                        state.last_positions = current
            except Exception as exc:
                logger.debug(
                    "Position reconciliation error",
                    operation="system_monitor_positions",
                    stage="run_loop",
                    error=str(exc),
                    exc_info=True,
                )
            await asyncio.sleep(interval_seconds)

    # ------------------------------------------------------------------
    async def _fetch_positions(self, bot: Any) -> list[Any]:
        try:
            broker_calls = getattr(getattr(bot, "context", None), "broker_calls", None)
            if broker_calls is not None and asyncio.iscoroutinefunction(
                getattr(broker_calls, "__call__", None)
            ):
                raw = await broker_calls(bot.broker.list_positions)
                if raw is None:
                    return []
                if isinstance(raw, list):
                    return raw
                try:
                    return list(raw)
                except TypeError:
                    return []
            return await asyncio.to_thread(bot.broker.list_positions)
        except Exception:
            return []

    def _normalize_positions(self, positions: list[Any]) -> dict[str, dict[str, str]]:
        normalized: dict[str, dict[str, str]] = {}
        for pos in positions or []:
            try:
                symbol = getattr(pos, "symbol", None)
                if not symbol:
                    continue
                quantity = quantity_from(pos) or Decimal("0")
                side = getattr(pos, "side", "")
                normalized[str(symbol)] = {
                    "quantity": str(quantity),
                    "side": str(side),
                }
            except Exception as exc:
                logger.exception(
                    "Failed to normalize position",
                    operation="system_monitor_positions",
                    stage="normalize",
                    symbol=str(getattr(pos, "symbol", "unknown")),
                    error=str(exc),
                )
        return normalized

    def _calculate_diff(
        self, previous: dict[str, dict[str, str]], current: dict[str, dict[str, str]]
    ) -> dict[str, dict[str, dict[str, str]]]:
        changes: dict[str, dict[str, dict[str, str]]] = {}
        for symbol, data in current.items():
            if previous.get(symbol) != data:
                changes[symbol] = {"old": previous.get(symbol, {}), "new": data}

        for symbol in previous:
            if symbol not in current:
                changes[symbol] = {"old": previous[symbol], "new": {}}

        return changes

    def _emit_position_changes(self, bot: Any, changes: dict[str, dict[str, Any]]) -> None:
        logger.info(
            "Position changes detected",
            operation="system_monitor_positions",
            stage="emit_changes",
            change_count=len(changes),
        )
        try:
            # Local import to avoid circular dependency with __init__.py
            from gpt_trader.monitoring.system import get_logger as _get_plog

            plog = _get_plog()
            for symbol, change in changes.items():
                new_data = change.get("new", {}) or {}
                size = float(new_data.get("quantity") or 0.0)
                side = str(new_data.get("side") or "")
                plog.log_position_change(symbol=symbol, side=side, size=size)
        except Exception as exc:
            logger.debug(
                "Failed to log position change metric",
                operation="system_monitor_positions",
                stage="emit_changes",
                error=str(exc),
                exc_info=True,
            )

        emit_metric(
            self._event_store,
            self._bot_id,
            {"event_type": "position_drift", "changes": changes},
            logger=logger,
        )
