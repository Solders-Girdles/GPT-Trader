from __future__ import annotations

import json
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping


class JsonFileStore:
    """Utility helper for JSON/JSONL persistence with shared locking semantics."""

    def __init__(
        self,
        path: Path,
        *,
        create: bool = True,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if create and not self.path.exists():
            self.path.touch()
        # Re-entrant lock so callers can compose operations safely
        self._lock = threading.RLock()

    def append_jsonl(self, payload: Mapping[str, Any]) -> None:
        """Append a JSON-serialisable mapping as a JSONL record."""
        line = json.dumps(dict(payload), default=self._default_serializer) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def iter_jsonl(self) -> Iterator[dict[str, Any]]:
        """Yield decoded JSON objects from a JSONL file."""
        if not self.path.exists():
            return
        with self._lock, self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(data, MutableMapping):
                    yield dict(data)

    def read_json(self, default: Any = None) -> Any:
        """Read a JSON document from disk, returning ``default`` on error."""
        try:
            with self._lock, self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return default
        except json.JSONDecodeError:
            return default

    def write_json(self, payload: Any, *, indent: int | None = 2) -> None:
        """Persist a JSON document atomically."""
        with self._lock, self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, default=self._default_serializer)
            if indent is not None:
                handle.write("\n")

    def replace_jsonl(self, entries: Iterable[Mapping[str, Any]]) -> None:
        """Rewrite the JSONL file with the provided entries."""
        with self._lock, self.path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(
                    json.dumps(dict(entry), default=self._default_serializer) + "\n"
                )

    @staticmethod
    def _default_serializer(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)


__all__ = ["JsonFileStore"]
