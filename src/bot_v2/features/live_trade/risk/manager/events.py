"""Event recording helpers for risk management."""

from __future__ import annotations

from typing import Any

from .logging import logger


class LiveRiskManagerEventMixin:
    """Provide structured event recording for risk workflows."""

    def _record_circuit_breaker_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Record circuit breaker related events into the event store."""
        store = getattr(self, "event_store", None)
        if store is None:
            return

        details = dict(payload)
        details.setdefault("reason", payload.get("reason", "unspecified"))
        details.setdefault("timestamp", self._now().isoformat())

        if hasattr(store, "store_event"):
            try:
                store.store_event(event_type, details)
                return
            except Exception:
                logger.debug("Failed to store_event for %s", event_type, exc_info=True)

        if hasattr(store, "append_error"):
            bot_id = details.get("bot_id") or getattr(self, "bot_id", "risk_manager")
            try:
                store.append_error(str(bot_id), message=event_type, context=details)
            except Exception:
                logger.debug("Failed to append_error for %s", event_type, exc_info=True)

    def _record_risk_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Record risk validation lifecycle events."""
        store = getattr(self, "event_store", None)
        if store is None or not hasattr(store, "store_event"):
            return
        try:
            store.store_event(event_type, dict(payload))
        except Exception:
            logger.debug("Failed to record %s event", event_type, exc_info=True)

    def handle_broker_error(self, error: Exception, context: dict[str, Any] | None = None) -> None:
        """Backward-compatible broker error handler retained for legacy orchestration flows."""
        logger.warning(
            "Handling broker error in risk manager",
            error=str(error),
            context=context or {},
            operation="risk_manager_handle_broker_error",
        )
        if self.event_store is not None and hasattr(self.event_store, "store_event"):
            payload = {"error": str(error), "context": context or {}}
            try:
                self.event_store.store_event("risk_manager_error", payload)
            except Exception:
                pass


__all__ = ["LiveRiskManagerEventMixin"]
