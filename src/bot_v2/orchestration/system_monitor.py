from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.config.path_registry import RUNTIME_DATA_DIR
from bot_v2.monitoring.system import LogLevel
from bot_v2.monitoring.system import get_logger as _get_plog

try:  # pragma: no cover - psutil optional
    from bot_v2.monitoring.system.collectors import ResourceCollector

    ResourceCollectorType: type[ResourceCollector] | None = ResourceCollector
except Exception:  # noqa: BLE001 - degrade gracefully when psutil missing
    ResourceCollectorType = None  # type: ignore[misc]
from bot_v2.orchestration.account_telemetry import AccountTelemetryService
from bot_v2.orchestration.config_controller import ConfigController
from bot_v2.orchestration.configuration import ConfigValidationError
from bot_v2.utilities.quantities import quantity_from

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Handles status logging, metrics publishing, and configuration change detection."""

    def __init__(
        self,
        bot: PerpsBot,
        account_telemetry: AccountTelemetryService | None = None,
    ) -> None:
        self._bot = bot
        self._account_telemetry = account_telemetry
        self._resource_collector = None
        if ResourceCollector is not None:
            try:
                self._resource_collector = ResourceCollector()
            except Exception as exc:  # pragma: no cover - psutil unavailable or restricted
                logger.debug("Resource collector unavailable: %s", exc, exc_info=True)

    def attach_account_telemetry(self, service: AccountTelemetryService) -> None:
        self._account_telemetry = service

    async def log_status(self) -> None:
        bot = self._bot
        positions = []
        balances = []
        try:
            positions = await asyncio.to_thread(bot.broker.list_positions)
        except Exception as e:
            logger.warning(f"Unable to fetch positions for status log: {e}")
        try:
            balances = await asyncio.to_thread(bot.broker.list_balances)
        except Exception as e:
            logger.warning(f"Unable to fetch balances for status log: {e}")

        usd_balance = next((b for b in balances if getattr(b, "asset", "").upper() == "USD"), None)
        equity = usd_balance.available if usd_balance else Decimal("0")

        logger.info("=" * 60)
        logger.info(
            f"Bot Status - {datetime.now()} - Profile: {bot.config.profile.value} - Equity: ${equity} - Positions: {len(positions)}"
        )
        for symbol, decision in bot.last_decisions.items():
            logger.info(f"  {symbol}: {decision.action.value} ({decision.reason})")
        logger.info("=" * 60)

        try:
            open_orders_count = len(bot.orders_store.get_open_orders())
        except Exception:
            open_orders_count = 0

        metrics_payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "profile": bot.config.profile.value,
            "equity": float(equity) if isinstance(equity, Decimal) else equity,
            "positions": [
                {
                    "symbol": getattr(p, "symbol", ""),
                    "quantity": float(quantity_val),
                    "side": getattr(p, "side", ""),
                    "entry_price": float(getattr(p, "entry_price", 0) or 0),
                    "mark_price": float(getattr(p, "mark_price", 0) or 0),
                }
                for p in positions
                if getattr(p, "symbol", None)
                for quantity_val in [quantity_from(p) or Decimal("0")]
            ],
            "decisions": {
                sym: {"action": decision.action.value, "reason": decision.reason}
                for sym, decision in bot.last_decisions.items()
            },
            "order_stats": dict(bot.order_stats),
            "open_orders": open_orders_count,
            "uptime_seconds": (datetime.now(UTC) - bot.start_time).total_seconds(),
        }

        if self._account_telemetry:
            snapshot = self._account_telemetry.latest_snapshot
            if snapshot:
                metrics_payload["account_snapshot"] = snapshot

        if self._resource_collector is not None:
            try:
                usage = self._resource_collector.collect()
                system_metrics: dict[str, float] = {
                    "cpu_percent": usage.cpu_percent,
                    "memory_percent": usage.memory_percent,
                    "memory_used_mb": usage.memory_mb,
                    "disk_percent": usage.disk_percent,
                    "disk_used_gb": usage.disk_gb,
                    "network_sent_mb": usage.network_sent_mb,
                    "network_recv_mb": usage.network_recv_mb,
                    "open_files": usage.open_files,
                    "threads": usage.threads,
                }
                metrics_payload["system"] = system_metrics
            except Exception as exc:
                logger.debug("Unable to collect system metrics: %s", exc, exc_info=True)

        self._publish_metrics(metrics_payload)

    def _publish_metrics(self, metrics: dict[str, Any]) -> None:
        bot = self._bot
        try:
            bot.event_store.append_metric(
                bot_id=bot.bot_id, metrics={"event_type": "cycle_metrics", **metrics}
            )
        except Exception as exc:
            logger.exception("Failed to persist cycle metrics: %s", exc)

        try:
            metrics_path = (
                RUNTIME_DATA_DIR / "perps_bot" / bot.config.profile.value / "metrics.json"
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to write metrics file: {e}")

        try:
            _get_plog().log_event(
                level=LogLevel.INFO,
                event_type="metrics_update",
                message="Cycle metrics updated",
                component="PerpsBot",
                **{k: v for k, v in metrics.items() if k not in {"positions", "decisions"}},
            )
        except Exception as exc:
            logger.debug("Failed to emit metrics update event: %s", exc, exc_info=True)

    def check_config_updates(self) -> None:
        bot = self._bot
        controller: ConfigController | None = getattr(bot, "config_controller", None)
        if controller is None:
            return
        try:
            change = controller.refresh_if_changed()
        except ConfigValidationError as exc:
            logger.error("Configuration update rejected: %s", exc)
            return

        if not change:
            return

        diff = change.diff
        if diff:
            logger.warning(
                "Configuration change detected for profile %s: %s",
                controller.current.profile.value,
                diff,
            )
        else:
            logger.warning(
                "Configuration inputs changed for profile %s; restart recommended to apply updates",
                controller.current.profile.value,
            )
        try:
            self._bot.apply_config_change(change)
        except Exception as exc:
            logger.exception("Failed to apply configuration change: %s", exc)
        finally:
            controller.consume_pending_change()

    def write_health_status(self, ok: bool, message: str = "", error: str = "") -> None:
        bot = self._bot
        status = {
            "ok": ok,
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "error": error,
        }
        status_file = RUNTIME_DATA_DIR / "perps_bot" / bot.config.profile.value / "health.json"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)

    async def run_position_reconciliation(self, interval_seconds: int = 90) -> None:
        await self._run_position_reconciliation_loop(interval_seconds=interval_seconds)

    async def _run_position_reconciliation_loop(self, interval_seconds: int = 90) -> None:
        bot = self._bot
        while bot.running:
            try:
                try:
                    positions = await asyncio.to_thread(bot.broker.list_positions)
                except Exception:
                    positions = []

                current: dict[str, dict[str, Any]] = {}
                for p in positions:
                    try:
                        sym = getattr(p, "symbol", None)
                        if not sym:
                            continue
                        quantity = quantity_from(p) or Decimal("0")
                        side = getattr(p, "side", "")
                        current[str(sym)] = {
                            "quantity": str(quantity),
                            "side": str(side),
                        }
                    except Exception as exc:
                        logger.exception(
                            "Failed to normalize position for %s: %s",
                            getattr(p, "symbol", "unknown"),
                            exc,
                            exc_info=True,
                        )
                        continue

                if not bot._last_positions and current:
                    bot._last_positions = current
                else:
                    changes: dict[str, dict[str, Any]] = {}
                    for sym, data in current.items():
                        prev = bot._last_positions.get(sym)
                        if (
                            not prev
                            or prev.get("quantity") != data.get("quantity")
                            or prev.get("side") != data.get("side")
                        ):
                            changes[sym] = {"old": prev or {}, "new": data}
                    for sym in list(bot._last_positions.keys()):
                        if sym not in current:
                            changes[sym] = {"old": bot._last_positions[sym], "new": {}}

                    if changes:
                        logger.info(f"Position changes detected: {len(changes)} updates")
                        try:
                            plog = _get_plog()
                            for sym, change in changes.items():
                                newd = change.get("new", {}) or {}
                                sided = newd.get("side")
                                qstr = newd.get("quantity")
                                sizef = float(qstr) if qstr and qstr not in ("", 0) else 0.0
                                plog.log_position_change(
                                    symbol=sym,
                                    side=str(sided) if sided is not None else "",
                                    size=sizef,
                                )
                        except Exception as exc:
                            logger.debug(
                                "Failed to log position change metric: %s", exc, exc_info=True
                            )
                        try:
                            bot.event_store.append_metric(
                                bot_id=bot.bot_id,
                                metrics={"event_type": "position_drift", "changes": changes},
                            )
                        except Exception as exc:
                            logger.exception("Failed to emit position drift metric: %s", exc)
                        bot._last_positions = current
            except Exception as e:
                logger.debug(f"Position reconciliation error: {e}", exc_info=True)
            await asyncio.sleep(interval_seconds)
