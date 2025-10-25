"""Reduce-only mode coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bot_v2.features.live_trade.risk import RiskRuntimeState
from bot_v2.orchestration.state_manager import ReduceOnlyModeSource
from bot_v2.utilities.telemetry import emit_metric

from .logging_utils import logger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .coordinator import RuntimeCoordinator


class RuntimeCoordinatorReduceOnlyMixin:
    """Manage reduce-only toggles and telemetry."""

    def set_reduce_only_mode(self: RuntimeCoordinator, enabled: bool, reason: str) -> None:
        controller = self._config_controller
        risk_manager = self.context.risk_manager

        state_manager = getattr(self.context.registry, "reduce_only_state_manager", None)
        if state_manager is not None:
            changed = state_manager.set_reduce_only_mode(
                enabled=enabled,
                reason=reason,
                source=ReduceOnlyModeSource.RUNTIME_COORDINATOR,
                metadata={"context": "runtime_coordinator"},
            )
            if changed:
                if controller is not None:
                    updated = controller.current.with_overrides(reduce_only_mode=enabled)
                    controller._set_current_config(updated)
                if risk_manager is not None:
                    risk_manager.set_reduce_only_mode(enabled, reason=reason)
                self._reduce_only_mode = bool(enabled)
                logger.warning(
                    "Reduce-only mode %s (%s)",
                    "enabled" if enabled else "disabled",
                    reason,
                    operation="reduce_only_toggle",
                    stage="set",
                    enabled=bool(enabled),
                    reason=reason,
                )
                self._emit_reduce_only_metric(enabled, reason)
            return

        if controller is None:
            self._reduce_only_mode = bool(enabled)
            if risk_manager is not None:
                try:
                    risk_manager.set_reduce_only_mode(enabled, reason=reason)
                except Exception:
                    logger.debug(
                        "Risk manager set_reduce_only_mode failed in legacy fallback",
                        operation="reduce_only_toggle",
                        stage="risk_manager_callback",
                        enabled=bool(enabled),
                        reason=reason,
                        exc_info=True,
                    )
            logger.debug(
                "No config controller available to toggle reduce-only mode",
                operation="reduce_only_toggle",
                stage="controller_missing",
            )
            self._emit_reduce_only_metric(enabled, reason)
            return

        self._legacy_toggle_reduce_only(controller, risk_manager, enabled, reason)

    def _legacy_toggle_reduce_only(
        self: RuntimeCoordinator,
        controller: Any,
        risk_manager: Any,
        enabled: bool,
        reason: str,
    ) -> None:
        if not controller.set_reduce_only_mode(enabled, reason=reason, risk_manager=risk_manager):
            return
        self._reduce_only_mode = bool(enabled)
        logger.warning(
            "Reduce-only mode %s (%s)",
            "enabled" if enabled else "disabled",
            reason,
            operation="reduce_only_toggle",
            stage="set",
            enabled=bool(enabled),
            reason=reason,
        )
        self._emit_reduce_only_metric(enabled, reason)

    def is_reduce_only_mode(self: RuntimeCoordinator) -> bool:
        state_manager = getattr(self.context.registry, "reduce_only_state_manager", None)
        if state_manager is not None:
            return state_manager.is_reduce_only_mode

        controller = self._config_controller
        if controller is None:
            return bool(self._reduce_only_mode)
        return bool(controller.is_reduce_only_mode(self.context.risk_manager))

    def on_risk_state_change(self: RuntimeCoordinator, state: RiskRuntimeState) -> None:
        state_manager = getattr(self.context.registry, "reduce_only_state_manager", None)
        if state_manager is not None:
            reduce_only = bool(state.reduce_only_mode)
            reason = state.last_reduce_only_reason or "unspecified"
            changed = state_manager.set_reduce_only_mode(
                enabled=reduce_only,
                reason=reason,
                source=ReduceOnlyModeSource.RISK_MANAGER,
                metadata={"context": "risk_state_change"},
            )
            if not changed:
                return
            controller = self._config_controller
            if controller is not None:
                controller.apply_risk_update(reduce_only)
            logger.warning(
                "Risk manager toggled reduce-only mode",
                enabled=reduce_only,
                reason=reason,
                operation="reduce_only_toggle",
                stage="risk_update",
            )
            self._emit_reduce_only_metric(reduce_only, reason)
            return

        controller = self._config_controller
        if controller is None:
            return
        reduce_only = bool(state.reduce_only_mode)
        if not controller.apply_risk_update(reduce_only):
            return
        reason = state.last_reduce_only_reason or "unspecified"
        logger.warning(
            "Risk manager toggled reduce-only mode",
            enabled=reduce_only,
            reason=reason,
            operation="reduce_only_toggle",
            stage="risk_update",
        )
        self._emit_reduce_only_metric(reduce_only, reason)

    def _emit_reduce_only_metric(self: RuntimeCoordinator, enabled: bool, reason: str) -> None:
        event_store = self.context.event_store
        if event_store is None:
            return
        emit_metric(
            event_store,
            self.context.bot_id,
            {
                "event_type": "reduce_only_mode_changed",
                "enabled": enabled,
                "reason": reason,
            },
            logger=logger,
        )


__all__ = ["RuntimeCoordinatorReduceOnlyMixin"]
