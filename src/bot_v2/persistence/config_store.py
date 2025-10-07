from __future__ import annotations

from pathlib import Path
from typing import Any

from bot_v2.persistence.json_file_store import JsonFileStore


class ConfigStore:
    """Persist bot definitions to a JSON file and load them on startup."""

    def __init__(self, root: Path | None = None) -> None:
        base = root or (Path(__file__).resolve().parents[3] / "results" / "managed")
        self.path = base / "bots.json"
        self._store = JsonFileStore(self.path)
        if self.path.stat().st_size == 0:
            self._store.write_json({"bots": []})

    def load_bots(self) -> list[dict[str, Any]]:
        data = self._store.read_json(default={"bots": []}) or {"bots": []}
        bots = data.get("bots") if isinstance(data, dict) else []
        if isinstance(bots, list):
            return list(bots)
        return []

    def save_bots(self, bots: list[dict[str, Any]]) -> None:
        payload = {"bots": bots}
        self._store.write_json(payload)

    def add_bot(self, config: dict[str, Any]) -> None:
        bots = self.load_bots()
        # replace if exists
        bots = [b for b in bots if b.get("bot_id") != config.get("bot_id")]
        bots.append(config)
        self.save_bots(bots)

    def remove_bot(self, bot_id: str) -> None:
        bots = self.load_bots()
        bots = [b for b in bots if b.get("bot_id") != bot_id]
        self.save_bots(bots)

    def update_bot(self, bot_id: str, updates: dict[str, Any]) -> None:
        bots = self.load_bots()
        out: list[dict[str, Any]] = []
        for b in bots:
            if b.get("bot_id") == bot_id:
                nb = dict(b)
                nb.update(updates)
                out.append(nb)
            else:
                out.append(b)
        self.save_bots(out)
