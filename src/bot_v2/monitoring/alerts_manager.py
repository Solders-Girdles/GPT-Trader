from __future__ import annotations

import asyncio
import logging
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

try:
    import yaml  # type: ignore
except Exception:
    yaml = cast(Any, None)

from bot_v2.monitoring.alerts import Alert, AlertDispatcher, AlertLevel

logger = logging.getLogger(__name__)


@dataclass
class AlertManager:
    """Unified alert manager with dispatching, history, and deduplication.

    Combines the functionality of:
    - AlertDispatcher for multi-channel routing (Slack, PagerDuty, Email, etc.)
    - Alert history and deduplication
    - YAML profile configuration support

    This replaces bot_v2.monitoring.system.alerting.AlertManager.
    """

    dispatcher: AlertDispatcher
    alert_history: list[Alert] = field(default_factory=list)
    max_history: int = 1000
    dedup_window_seconds: int = 300  # 5 minutes

    # Deduplication tracking
    _recent_alerts: dict[str, datetime] = field(default_factory=dict, init=False)

    # Profile resolution fallbacks
    _PROFILE_FALLBACKS = {
        "canary": Path("config/profiles/canary.yaml"),
        "spot": Path("config/profiles/spot.yaml"),
        "dev": Path("config/profiles/dev_entry.yaml"),
        "demo": Path("config/profiles/dev_entry.yaml"),
        "prod": Path("config/profiles/canary.yaml"),
    }

    @classmethod
    def _load_dispatcher_config(
        cls,
        *,
        path: str | Path | None,
        profile: str | Any | None,
    ) -> dict[str, Any]:
        if yaml is None:
            return {}

        resolved_path = cls._resolve_config_path(path=path, profile=profile)
        if resolved_path is None:
            return {}

        try:
            with resolved_path.open() as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception as exc:  # pragma: no cover - filesystem/env issues
            logger.debug("Failed to load alert config from %s: %s", resolved_path, exc)
            return {}

        if not isinstance(payload, Mapping):
            return {}

        monitoring_section = payload.get("monitoring")
        if not isinstance(monitoring_section, Mapping):
            return {}

        alerts_config = monitoring_section.get("alerts")
        if not isinstance(alerts_config, Mapping):
            return {}

        if alerts_config.get("enabled") is False:
            return {}

        return cls._map_channel_config(alerts_config)

    @classmethod
    def _resolve_config_path(
        cls,
        *,
        path: str | Path | None,
        profile: str | Any | None,
    ) -> Path | None:
        candidates: list[Path] = []

        def _as_path(candidate: str | Path | None) -> Path | None:
            if not candidate:
                return None
            try:
                return Path(str(candidate)).expanduser()
            except Exception:  # pragma: no cover - defensive
                return None

        explicit = _as_path(path)
        if explicit is not None:
            candidates.append(explicit)

        env_override = _as_path(os.getenv("PERPS_ALERT_CONFIG"))
        if env_override is not None:
            candidates.append(env_override)

        resolved_profile = cls._normalize_profile(profile)
        if resolved_profile is None:
            resolved_profile = cls._normalize_profile(os.getenv("PERPS_PROFILE"))

        if resolved_profile:
            fallback = cls._PROFILE_FALLBACKS.get(resolved_profile)
            if fallback is not None:
                candidates.append(fallback)
            candidates.append(Path(f"config/profiles/{resolved_profile}.yaml"))

        # Ensure canary remains last-resort default
        candidates.append(cls._PROFILE_FALLBACKS["canary"])

        for candidate in candidates:
            try:
                if candidate.exists():
                    return candidate
            except Exception:  # pragma: no cover - defensive
                continue
        return None

    @staticmethod
    def _normalize_profile(profile: Any | None) -> str | None:
        if profile is None:
            return None
        if hasattr(profile, "value"):
            profile = getattr(profile, "value")
        value = str(profile).strip().lower()
        return value or None

    @classmethod
    def _map_channel_config(cls, alerts_config: Mapping[str, Any]) -> dict[str, Any]:
        dispatcher_config: dict[str, Any] = {}
        raw_channels = alerts_config.get("channels")
        channels: Iterable[Mapping[str, Any]]
        if isinstance(raw_channels, list):
            channels = [c for c in raw_channels if isinstance(c, Mapping)]
        elif isinstance(raw_channels, Mapping):
            channels = [raw_channels]
        else:
            channels = []

        for channel in channels:
            channel_type = str(channel.get("type") or "").strip().lower()
            severity = str(channel.get("level") or channel.get("severity") or "WARNING").upper()

            if channel_type == "slack" and "slack_webhook_url" not in dispatcher_config:
                webhook = cls._expand_env_var(channel.get("webhook_url"))
                if webhook:
                    dispatcher_config["slack_webhook_url"] = webhook
                    dispatcher_config["slack_min_severity"] = severity
            elif channel_type == "pagerduty" and "pagerduty_api_key" not in dispatcher_config:
                api_key = cls._expand_env_var(channel.get("api_key"))
                if api_key:
                    dispatcher_config["pagerduty_api_key"] = api_key
                    dispatcher_config["pagerduty_min_severity"] = severity
            elif channel_type == "email" and "email" not in dispatcher_config:
                email_cfg = cls._build_email_config(channel, severity)
                if email_cfg:
                    dispatcher_config["email"] = email_cfg

        return dispatcher_config

    @staticmethod
    def _expand_env_var(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        expanded = os.path.expandvars(value)
        if expanded == value and ("${" in value or "$" in value):
            return None
        expanded = expanded.strip()
        return expanded or None

    @classmethod
    def _build_email_config(
        cls, channel: Mapping[str, Any], severity: str
    ) -> dict[str, Any] | None:
        host = cls._expand_env_var(channel.get("smtp_host"))
        from_email = cls._expand_env_var(channel.get("from_email"))
        if not host or not from_email:
            return None

        port_raw = channel.get("smtp_port")
        try:
            port = int(port_raw) if port_raw is not None else 587
        except (TypeError, ValueError):
            logger.debug("Invalid SMTP port in alert config: %s", port_raw)
            return None

        to_raw = channel.get("to_emails") or channel.get("recipients")
        to_emails: list[str] = []
        if isinstance(to_raw, str):
            tokens = [t.strip() for t in to_raw.replace(";", ",").split(",") if t.strip()]
            to_emails = tokens
        elif isinstance(to_raw, list):
            to_emails = [str(addr).strip() for addr in to_raw if str(addr).strip()]

        if not to_emails:
            return None

        username = cls._expand_env_var(channel.get("username"))
        password = cls._expand_env_var(channel.get("password"))
        use_tls = channel.get("use_tls")
        if isinstance(use_tls, str):
            use_tls = use_tls.strip().lower() in {"1", "true", "yes", "on"}
        elif not isinstance(use_tls, bool):
            use_tls = True

        return {
            "smtp_host": host,
            "smtp_port": port,
            "from_email": from_email,
            "to_emails": to_emails,
            "username": username,
            "password": password,
            "use_tls": use_tls,
            "min_severity": severity,
        }

    def _dispatch_alert(self, alert: Alert) -> None:
        if not self.dispatcher:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(self.dispatcher.dispatch(alert))
            except Exception as exc:  # pragma: no cover - logging best effort
                logger.debug("Failed to dispatch alert (fresh loop): %s", exc, exc_info=True)
            return

        try:
            loop.create_task(self.dispatcher.dispatch(alert))
        except RuntimeError:
            try:
                asyncio.run(self.dispatcher.dispatch(alert))
            except Exception as exc:  # pragma: no cover - best effort logging
                logger.debug("Failed to dispatch alert via asyncio.run: %s", exc, exc_info=True)

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any] | None) -> AlertManager:
        """Instantiate an alert manager from alerts settings."""

        if not isinstance(settings, Mapping) or settings.get("enabled") is False:
            return cls(dispatcher=AlertDispatcher.from_config({}))

        dispatcher_config = cls._map_channel_config(settings)
        return cls(dispatcher=AlertDispatcher.from_config(dispatcher_config))

    @classmethod
    def from_profile_yaml(
        cls,
        path: str | Path | None = None,
        *,
        profile: str | Any | None = None,
    ) -> AlertManager:
        """Instantiate alert manager using alert config derived from a profile file.

        Args:
            path: Explicit path to profile YAML (takes precedence when provided)
            profile: Optional profile name/enum used to resolve default profile file

        Returns:
            Configured ``AlertManager`` instance
        """

        config = cls._load_dispatcher_config(path=path, profile=profile)
        return cls(dispatcher=AlertDispatcher.from_config(config))

    def create_alert(
        self,
        level: AlertLevel | str,
        source: str,
        message: str,
        title: str | None = None,
        context: dict[str, Any] | None = None,
        *,
        dispatch: bool = True,
    ) -> Alert | None:
        """Create an alert with deduplication.

        Args:
            level: Alert severity level (AlertLevel enum or string)
            source: Alert source/component name
            message: Alert message
            title: Optional alert title (defaults to "{source} Alert")
            context: Additional context dictionary
            dispatch: Whether to dispatch the alert immediately via dispatcher

        Returns:
            Alert object or None if deduplicated
        """
        # Convert string level to AlertLevel
        if isinstance(level, str):
            level = AlertLevel[level.upper()]

        current_time = datetime.now()

        # Check for duplicate alerts
        alert_key = f"{source}:{message}"
        if alert_key in self._recent_alerts:
            time_since = (current_time - self._recent_alerts[alert_key]).total_seconds()
            if time_since < 0:
                # Clock moved backwards (e.g., FrozenTime), reset tracking entry
                self._recent_alerts.pop(alert_key, None)
            elif time_since <= self.dedup_window_seconds:
                # Duplicate alert within window, skip
                return None

        # Create alert
        alert_id = str(uuid.uuid4())[:8]
        alert = Alert(
            timestamp=current_time,
            source=source,
            severity=level,
            title=title or f"{source} Alert",
            message=message,
            context=context or {},
            alert_id=alert_id,
        )

        # Store in history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        # Update deduplication tracking
        self._recent_alerts[alert_key] = current_time

        if dispatch:
            # Dispatch alert asynchronously (best effort)
            self._dispatch_alert(alert)

        return alert

    def get_recent_alerts(
        self, count: int = 10, severity: AlertLevel | None = None, source: str | None = None
    ) -> list[Alert]:
        """Get recent alerts with optional filtering.

        Args:
            count: Number of recent alerts to return
            severity: Optional severity filter
            source: Optional source filter

        Returns:
            List of recent alerts
        """
        alerts = list(self.alert_history)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]

        return alerts[-count:]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get summary of recent alerts.

        Returns:
            Dictionary with alert counts by severity and sources
        """
        recent = self.alert_history[-100:]  # Last 100 alerts

        summary: dict[str, Any] = {
            "total": len(recent),
            "by_severity": {
                "critical": len([a for a in recent if a.severity == AlertLevel.CRITICAL]),
                "error": len([a for a in recent if a.severity == AlertLevel.ERROR]),
                "warning": len([a for a in recent if a.severity == AlertLevel.WARNING]),
                "info": len([a for a in recent if a.severity == AlertLevel.INFO]),
            },
            "sources": list({a.source for a in recent}),
        }

        return summary

    def cleanup_old_alerts(self, retention_hours: int = 24) -> None:
        """Clean up old alerts from history.

        Args:
            retention_hours: Keep alerts newer than this many hours
        """
        current_time = datetime.now()
        cutoff = current_time - timedelta(hours=retention_hours)
        self.alert_history = [
            alert for alert in self.alert_history if cutoff < alert.timestamp <= current_time
        ]

        # Clean up deduplication tracking
        self._recent_alerts = {
            key: timestamp
            for key, timestamp in self._recent_alerts.items()
            if 0 <= (current_time - timestamp).total_seconds() <= self.dedup_window_seconds
        }

    # Backward compatibility methods for system/alerting.py API
    def get_all_alerts(self) -> list[Alert]:
        """Get all alerts (alias for alert_history).

        For backward compatibility with monitoring.system.alerting.AlertManager.
        """
        return self.alert_history

    def get_active_alerts(self) -> list[Alert]:
        """Get recent active alerts.

        For backward compatibility with monitoring.system.alerting.AlertManager.
        Returns alerts from the last hour.
        """
        cutoff = datetime.now() - timedelta(hours=1)
        return [a for a in self.alert_history if a.timestamp > cutoff]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        For backward compatibility with monitoring.system.alerting.AlertManager.
        The new AlertManager doesn't track acknowledgment state, so this is a no-op.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if alert exists in history
        """
        return any(a.alert_id == alert_id for a in self.alert_history)

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        For backward compatibility with monitoring.system.alerting.AlertManager.
        The new AlertManager doesn't track resolution state, so this is a no-op.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if alert exists in history
        """
        return any(a.alert_id == alert_id for a in self.alert_history)
