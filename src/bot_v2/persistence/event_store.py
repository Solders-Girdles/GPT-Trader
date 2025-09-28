from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config.path_registry import DEFAULT_EVENT_STORE_DIR


class EventStore:
    """Lightweight JSONL event store for trades, positions, metrics, and errors.

    Writes one JSON object per line with at minimum: time, bot_id, type.
    File default: var/data/perps_bot/shared/events.jsonl (created if missing).
    Callers typically provide a profile-specific root such as
    var/data/perps_bot/<profile>/events.jsonl.
    """

    def __init__(self, root: Path | None = None):
        base = root or DEFAULT_EVENT_STORE_DIR
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / "events.jsonl"
        # Ensure file exists
        if not self.path.exists():
            self.path.touch()
        self._lock = threading.Lock()

    def _write(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("time", datetime.utcnow().isoformat())
        line = json.dumps(payload, default=self._default) + "\n"
        with self._lock, self.path.open("a") as f:
            f.write(line)

    @staticmethod
    def _default(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        return str(obj)

    # Public appenders
    def append_trade(self, bot_id: str, trade: dict[str, Any]) -> None:
        self._write({"type": "trade", "bot_id": bot_id, **trade})

    def append_position(self, bot_id: str, position: dict[str, Any]) -> None:
        self._write({"type": "position", "bot_id": bot_id, **position})

    def append_metric(self, bot_id: str, metrics: dict[str, Any]) -> None:
        self._write({"type": "metric", "bot_id": bot_id, **metrics})

    def append_error(
        self, bot_id: str, message: str, context: dict[str, Any] | None = None
    ) -> None:
        payload: dict[str, Any] = {"type": "error", "bot_id": bot_id, "message": message}
        if context:
            payload.update(context)
        self._write(payload)

    # Query helpers
    def tail(
        self, bot_id: str, limit: int = 50, types: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        types_set = set(types or [])
        out: list[dict[str, Any]] = []
        try:
            with self.path.open("r") as f:
                for line in f:
                    try:
                        evt = json.loads(line)
                    except Exception:
                        continue
                    if evt.get("bot_id") != bot_id:
                        continue
                    if types_set and evt.get("type") not in types_set:
                        continue
                    out.append(evt)
            return out[-limit:]
        except Exception:
            return []
