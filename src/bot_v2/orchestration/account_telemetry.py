"""Account telemetry collection for live trading sessions."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from bot_v2.config.path_registry import RUNTIME_DATA_DIR
from bot_v2.utilities.telemetry import emit_metric

logger = logging.getLogger(__name__)


class AccountTelemetryService:
    """Collects periodic account snapshots and persists them to storage."""

    _REQUIRED_BROKER_ATTRS = (
        "get_key_permissions",
        "get_fee_schedule",
        "get_account_limits",
        "get_transaction_summary",
        "list_payment_methods",
        "list_portfolios",
    )

    def __init__(
        self,
        *,
        broker: Any,
        account_manager: Any,
        event_store: Any,
        bot_id: str,
        profile: str,
    ) -> None:
        self._broker = broker
        self._account_manager = account_manager
        self._event_store = event_store
        self._bot_id = bot_id
        self._profile = profile
        self._latest_snapshot: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def supports_snapshots(self) -> bool:
        return all(hasattr(self._broker, attr) for attr in self._REQUIRED_BROKER_ATTRS)

    @property
    def latest_snapshot(self) -> dict[str, Any]:
        return dict(self._latest_snapshot)

    # ------------------------------------------------------------------
    def update_profile(self, profile: str) -> None:
        self._profile = profile

    # ------------------------------------------------------------------
    async def run(self, interval_seconds: int = 300) -> None:
        if not self.supports_snapshots():
            logger.info("Account snapshot telemetry disabled; broker lacks required endpoints")
            return
        while True:
            try:
                snapshot = await asyncio.to_thread(self.collect_snapshot)
                if snapshot:
                    self._publish_snapshot(snapshot)
            except Exception as exc:
                logger.debug("Account telemetry error: %s", exc, exc_info=True)
            await asyncio.sleep(interval_seconds)

    # ------------------------------------------------------------------
    def collect_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        try:
            snapshot.update(self._account_manager.snapshot(emit_metric=False))
        except Exception as exc:
            logger.debug("Failed to capture account manager snapshot: %s", exc, exc_info=True)
        try:
            server_time = self._broker.get_server_time()
            snapshot["server_time"] = server_time.isoformat() if server_time else None
        except Exception:
            snapshot.setdefault("server_time", None)
        snapshot["timestamp"] = datetime.now(UTC).isoformat()
        self._latest_snapshot = snapshot
        return snapshot

    # ------------------------------------------------------------------
    def _publish_snapshot(self, snapshot: dict[str, Any]) -> None:
        emit_metric(
            self._event_store,
            self._bot_id,
            {"event_type": "account_snapshot", **snapshot},
            logger=logger,
        )
        try:
            output_path = RUNTIME_DATA_DIR / "perps_bot" / self._profile / "account.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as fh:
                json.dump(snapshot, fh, indent=2)
        except Exception as exc:
            logger.debug("Failed to write account snapshot: %s", exc, exc_info=True)
