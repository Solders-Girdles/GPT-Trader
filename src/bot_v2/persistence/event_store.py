from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class EventStore:
    """Lightweight JSONL event store for trades, positions, metrics, and errors.

    Writes one JSON object per line with at minimum: time, bot_id, type.
    File: results/managed/events.jsonl (created if missing).
    """

    def __init__(self, root: Optional[Path] = None):
        base = root or (Path(__file__).resolve().parents[3] / 'results' / 'managed')
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / 'events.jsonl'
        # Ensure file exists
        if not self.path.exists():
            self.path.touch()
        self._lock = threading.Lock()

    def _write(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault('time', datetime.utcnow().isoformat())
        line = json.dumps(payload, default=self._default) + "\n"
        with self._lock, self.path.open('a') as f:
            f.write(line)

    @staticmethod
    def _default(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        return str(obj)

    # Public appenders
    def append_trade(self, bot_id: str, trade: Dict[str, Any]) -> None:
        self._write({'type': 'trade', 'bot_id': bot_id, **trade})

    def append_position(self, bot_id: str, position: Dict[str, Any]) -> None:
        self._write({'type': 'position', 'bot_id': bot_id, **position})

    def append_metric(self, bot_id: str, metrics: Dict[str, Any]) -> None:
        self._write({'type': 'metric', 'bot_id': bot_id, **metrics})

    def append_error(self, bot_id: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {'type': 'error', 'bot_id': bot_id, 'message': message}
        if context:
            payload.update(context)
        self._write(payload)

    # Query helpers
    def tail(self, bot_id: str, limit: int = 50, types: Optional[Iterable[str]] = None) -> list[Dict[str, Any]]:
        types_set = set(types or [])
        out: list[Dict[str, Any]] = []
        try:
            with self.path.open('r') as f:
                for line in f:
                    try:
                        evt = json.loads(line)
                    except Exception:
                        continue
                    if evt.get('bot_id') != bot_id:
                        continue
                    if types_set and evt.get('type') not in types_set:
                        continue
                    out.append(evt)
            return out[-limit:]
        except Exception:
            return []

