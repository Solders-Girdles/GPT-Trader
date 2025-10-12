from __future__ import annotations

import contextlib
import json
import os
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from bot_v2.config.path_registry import DEFAULT_EVENT_STORE_DIR
from bot_v2.persistence.json_file_store import JsonFileStore
from bot_v2.utilities import utc_now_iso


def _ensure_mapping(kind: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{kind} payload must be a mapping")
    return dict(payload)


def _require_non_empty_str(data: Mapping[str, Any], key: str, kind: str) -> str:
    value = data.get(key)
    if value is None:
        raise ValueError(f"{kind} payload missing '{key}'")
    value_str = str(value).strip()
    if not value_str:
        raise ValueError(f"{kind} payload has empty '{key}'")
    return value_str


def _stringify(value: Any) -> str:
    return str(value)


def _normalize_trade_payload(trade: Mapping[str, Any]) -> dict[str, Any]:
    payload = _ensure_mapping("trade", trade)
    payload.pop("bot_id", None)
    payload.pop("type", None)

    payload["symbol"] = _require_non_empty_str(payload, "symbol", "trade")
    payload["side"] = _require_non_empty_str(payload, "side", "trade").lower()

    quantity_value = payload.get("quantity", payload.get("size"))
    if quantity_value is None:
        raise ValueError("Trade payload missing 'quantity' or 'size'")
    payload["quantity"] = _stringify(quantity_value)

    if "status" in payload and payload["status"] is not None:
        payload["status"] = _stringify(payload["status"])
    if "order_id" in payload and payload["order_id"] is not None:
        payload["order_id"] = _stringify(payload["order_id"])
    if "client_order_id" in payload and payload["client_order_id"] is not None:
        payload["client_order_id"] = _stringify(payload["client_order_id"])
    if "price" in payload and payload["price"] is not None:
        payload["price"] = _stringify(payload["price"])

    return payload


def _normalize_position_payload(position: Mapping[str, Any]) -> dict[str, Any]:
    payload = _ensure_mapping("position", position)
    payload.pop("bot_id", None)
    payload.pop("type", None)

    payload["symbol"] = _require_non_empty_str(payload, "symbol", "position")
    quantity_value = payload.get("quantity", payload.get("size"))
    if quantity_value is None:
        raise ValueError("Position payload missing 'quantity' or 'size'")
    payload["quantity"] = _stringify(quantity_value)
    payload.pop("size", None)

    payload["mark_price"] = _require_non_empty_str(payload, "mark_price", "position")

    for key in ("entry_price", "unrealized_pnl", "realized_pnl", "position_value", "timestamp"):
        if key in payload and payload[key] is not None:
            payload[key] = _stringify(payload[key])

    if "side" in payload and payload["side"] is not None:
        payload["side"] = _stringify(payload["side"])

    return payload


def _normalize_metric_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    payload = _ensure_mapping("metric", metrics)
    payload.pop("bot_id", None)

    event_type = payload.get("event_type") or payload.get("type")
    if event_type is None:
        raise ValueError("Metric payload missing 'event_type'")
    event_type_str = _stringify(event_type)
    payload["event_type"] = event_type_str
    payload.setdefault("type", event_type_str)

    return payload


def _normalize_error_payload(message: str, context: Mapping[str, Any] | None) -> dict[str, Any]:
    message_str = str(message).strip()
    if not message_str:
        raise ValueError("Error payload requires a non-empty message")

    payload: dict[str, Any] = {"message": message_str}
    if context:
        context_payload = _ensure_mapping("error context", context)
        context_payload.pop("bot_id", None)
        context_payload.pop("type", None)
        payload.update(context_payload)
    return payload


class EventStore:
    """Lightweight JSONL event store for trades, positions, metrics, and errors.

    Writes one JSON object per line with at minimum: time, bot_id, type.
    File default: var/data/perps_bot/shared/events.jsonl (created if missing).
    Callers typically provide a profile-specific root such as
    var/data/perps_bot/<profile>/events.jsonl.
    """

    def __init__(self, root: Path | None = None) -> None:
        base = root or DEFAULT_EVENT_STORE_DIR
        self.path = base / "events.jsonl"
        self._store = JsonFileStore(self.path)
        self._lock = threading.RLock()

    def _write(self, payload: dict[str, Any]) -> None:
        record = self._normalize_payload(payload)
        with self._lock:
            self._store.append_jsonl(record)

    def _normalize_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if "bot_id" not in payload:
            raise ValueError("Event payload missing 'bot_id'")
        if "type" not in payload:
            raise ValueError("Event payload missing 'type'")

        record = dict(payload)
        record["bot_id"] = str(record["bot_id"])
        record["type"] = str(record["type"])
        record.setdefault("time", utc_now_iso())
        return record

    # Public appenders
    def append_trade(self, bot_id: str, trade: dict[str, Any]) -> None:
        payload = _normalize_trade_payload(trade)
        self._write({"type": "trade", "bot_id": bot_id, **payload})

    def append_position(self, bot_id: str, position: dict[str, Any]) -> None:
        payload = _normalize_position_payload(position)
        self._write({"type": "position", "bot_id": bot_id, **payload})

    def append_metric(self, bot_id: str, metrics: dict[str, Any]) -> None:
        payload = _normalize_metric_payload(metrics)
        self._write({"type": "metric", "bot_id": bot_id, **payload})

    def append_error(
        self, bot_id: str, message: str, context: dict[str, Any] | None = None
    ) -> None:
        payload = _normalize_error_payload(message, context)
        self._write({"type": "error", "bot_id": bot_id, **payload})

    # Query helpers
    def tail(
        self, bot_id: str, limit: int = 50, types: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        types_set = set(types or [])
        path = self._store.path
        if not path.exists():
            return []

        results: list[dict[str, Any]] = []

        try:
            lock = getattr(self._store, "_lock", None)
            lock_cm = lock if lock is not None else contextlib.nullcontext()
            with lock_cm:
                with path.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    buffer = bytearray()
                    position = handle.tell()
                    while position > 0 and len(results) < limit:
                        read_size = min(4096, position)
                        position -= read_size
                        handle.seek(position)
                        chunk = handle.read(read_size)
                        buffer[:0] = chunk

                        while True:
                            newline_index = buffer.rfind(b"\n")
                            if newline_index == -1:
                                break
                            line = buffer[newline_index + 1 :]
                            buffer = buffer[:newline_index]
                            if not line:
                                continue
                            event = self._decode_line(line)
                            if event is None:
                                continue
                            if event.get("bot_id") != bot_id:
                                continue
                            if types_set and event.get("type") not in types_set:
                                continue
                            results.append(event)
                            if len(results) == limit:
                                break

                    if len(results) < limit and buffer:
                        event = self._decode_line(buffer)
                        if (
                            event is not None
                            and event.get("bot_id") == bot_id
                            and (not types_set or event.get("type") in types_set)
                        ):
                            results.append(event)
        except Exception:
            return []

        results.reverse()
        return results

    @staticmethod
    def _decode_line(raw_line: bytes) -> dict[str, Any] | None:
        try:
            decoded = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
        if not decoded:
            return None
        try:
            event = json.loads(decoded)
        except json.JSONDecodeError:
            return None
        if not isinstance(event, dict):
            return None
        return event
