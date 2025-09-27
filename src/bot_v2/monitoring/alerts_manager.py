from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from .alerts import AlertDispatcher, AlertSeverity, Alert, create_system_alert

logger = logging.getLogger(__name__)


@dataclass
class AlertManager:
    """Thin wrapper around AlertDispatcher using profile YAML for config."""

    dispatcher: AlertDispatcher

    @classmethod
    def from_profile_yaml(cls, path: str = "config/profiles/canary.yaml") -> "AlertManager":
        config: Dict = {}
        if yaml is not None:
            try:
                with open(path, 'r') as f:
                    y = yaml.safe_load(f) or {}
                alerts_cfg = (y.get('monitoring') or {}).get('alerts') or {}
                # Map profile fields to dispatcher config format
                mapped: Dict = {
                    'slack_webhook_url': None,
                    'slack_min_severity': 'WARNING',
                    'pagerduty_api_key': None,
                    'pagerduty_min_severity': 'ERROR',
                }
                if isinstance(alerts_cfg, dict):
                    chans = alerts_cfg.get('channels') or []
                    for c in chans:
                        ctype = (c.get('type') or '').lower()
                        level = (c.get('level') or 'WARNING').upper()
                        if ctype == 'slack' and not mapped['slack_webhook_url']:
                            mapped['slack_webhook_url'] = c.get('webhook_url')
                            mapped['slack_min_severity'] = level
                        elif ctype == 'pagerduty' and not mapped['pagerduty_api_key']:
                            mapped['pagerduty_api_key'] = c.get('api_key')
                            mapped['pagerduty_min_severity'] = level
                config = mapped
            except Exception as e:
                logger.debug(f"Failed to load alert config from YAML: {e}")
        return cls(dispatcher=AlertDispatcher.from_config(config))

    async def send_alert(self, level: str, title: str, message: str, metrics: Optional[Dict] = None):
        sev = AlertSeverity[level.upper()] if level else AlertSeverity.WARNING
        alert = Alert(
            timestamp=datetime.utcnow(),
            source="perps_bot",
            severity=sev,
            title=title,
            message=message,
            context=metrics or {},
        )
        await self.dispatcher.dispatch(alert)

