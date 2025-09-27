from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .bot_manager import ManagedBot


class SessionStore:
    def __init__(self, root: Path | None = None):
        self.root = root or (Path(__file__).resolve().parents[3] / 'results' / 'managed')
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bot_snapshot(self, bot: ManagedBot) -> Path:
        stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        fn = self.root / f"bot_{bot.config.bot_id}_{stamp}.json"
        payload: Dict[str, Any] = {
            'config': asdict(bot.config),
            'status': bot.status,
            'metrics': asdict(bot.metrics),
            'error': bot.error,
        }
        with open(fn, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        return fn

