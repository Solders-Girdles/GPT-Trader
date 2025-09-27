from __future__ import annotations

import json
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigStore:
    """Persist bot definitions to a JSON file and load them on startup."""

    def __init__(self, root: Optional[Path] = None):
        base = root or (Path(__file__).resolve().parents[3] / 'results' / 'managed')
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / 'bots.json'
        self._lock = threading.Lock()
        if not self.path.exists():
            with self.path.open('w') as f:
                json.dump({'bots': []}, f, indent=2)

    def load_bots(self) -> List[Dict[str, Any]]:
        try:
            with self.path.open('r') as f:
                data = json.load(f)
            return list(data.get('bots') or [])
        except Exception:
            return []

    def save_bots(self, bots: List[Dict[str, Any]]) -> None:
        payload = {'bots': bots}
        with self._lock:
            with self.path.open('w') as f:
                json.dump(payload, f, indent=2)

    def add_bot(self, cfg: Dict[str, Any]) -> None:
        bots = self.load_bots()
        # replace if exists
        bots = [b for b in bots if b.get('bot_id') != cfg.get('bot_id')]
        bots.append(cfg)
        self.save_bots(bots)

    def remove_bot(self, bot_id: str) -> None:
        bots = self.load_bots()
        bots = [b for b in bots if b.get('bot_id') != bot_id]
        self.save_bots(bots)

    def update_bot(self, bot_id: str, updates: Dict[str, Any]) -> None:
        bots = self.load_bots()
        out: List[Dict[str, Any]] = []
        for b in bots:
            if b.get('bot_id') == bot_id:
                nb = dict(b)
                nb.update(updates)
                out.append(nb)
            else:
                out.append(b)
        self.save_bots(out)

