from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

try:
    import yaml  # type: ignore
except Exception:
    yaml = cast(Any, None)

try:
    from bot_v2.monitoring.alerts import AlertDispatcher
except ImportError:  # pragma: no cover - archive fallback
    from archived.experimental.monitoring.alerts import AlertDispatcher  # type: ignore

from bot_v2.monitoring.alert_types import Alert, AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class AlertManager:
    """Thin wrapper around AlertDispatcher using profile YAML for config."""

    dispatcher: AlertDispatcher

    @classmethod
    def from_profile_yaml(cls, path: str | Path = "config/profiles/canary.yaml") -> AlertManager:
        config: dict[str, Any] = {}
        if yaml is not None:
            try:
                path_obj = Path(path)
                with path_obj.open() as f:
                    loaded = yaml.safe_load(f) or {}
                data = loaded if isinstance(loaded, dict) else {}
                monitoring_section = data.get("monitoring")
                alerts_config = (
                    monitoring_section.get("alerts") if isinstance(monitoring_section, dict) else {}
                )
                # Map profile fields to dispatcher config format
                mapped: dict[str, Any] = {
                    "slack_webhook_url": None,
                    "slack_min_severity": "WARNING",
                    "pagerduty_api_key": None,
                    "pagerduty_min_severity": "ERROR",
                }
                if isinstance(alerts_config, dict):
                    chans = alerts_config.get("channels") or []
                    if not isinstance(chans, list):
                        chans = [chans]
                    for c in chans:
                        if not isinstance(c, Mapping):
                            continue
                        ctype = str(c.get("type") or "").lower()
                        level = str(c.get("level") or "WARNING").upper()
                        if ctype == "slack" and not mapped["slack_webhook_url"]:
                            mapped["slack_webhook_url"] = c.get("webhook_url")
                            mapped["slack_min_severity"] = level
                        elif ctype == "pagerduty" and not mapped["pagerduty_api_key"]:
                            mapped["pagerduty_api_key"] = c.get("api_key")
                            mapped["pagerduty_min_severity"] = level
                config = mapped
            except Exception as e:
                logger.debug(f"Failed to load alert config from YAML: {e}")
        return cls(dispatcher=AlertDispatcher.from_config(config))

    async def send_alert(
        self, level: str, title: str, message: str, metrics: dict[str, Any] | None = None
    ) -> None:
        sev = AlertSeverity[level.upper()] if level else AlertSeverity.WARNING
        alert = Alert(
            source="perps_bot",
            severity=sev,
            title=title,
            message=message,
            context=dict(metrics or {}),
        )
        await self.dispatcher.dispatch(alert)
