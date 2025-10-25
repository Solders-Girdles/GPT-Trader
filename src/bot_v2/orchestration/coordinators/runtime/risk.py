"""Risk manager bootstrap helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.orchestration.configuration import DEFAULT_SPOT_RISK_PATH, Profile, RiskConfig

from .logging_utils import logger

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.coordinators.base import CoordinatorContext

    from .coordinator import RuntimeCoordinator


class RuntimeCoordinatorRiskMixin:
    """Encapsulate risk manager initialization logic."""

    def _init_risk_manager(
        self: RuntimeCoordinator,
        context: CoordinatorContext | None = None,
    ) -> CoordinatorContext:
        ctx = context or self.context
        controller = self._config_controller

        if ctx.registry.risk_manager is not None:
            risk_manager = ctx.registry.risk_manager
            risk_manager.set_state_listener(self.on_risk_state_change)
            if controller is not None:
                controller.sync_with_risk_manager(risk_manager)
                risk_manager.set_reduce_only_mode(controller.reduce_only_mode, reason="config_init")
            return ctx.with_updates(risk_manager=risk_manager)

        settings, ctx = self._resolve_settings(ctx)
        env_risk_config_path = settings.risk_config_path
        resolved_risk_path = str(env_risk_config_path) if env_risk_config_path else None

        config_profile = ctx.config.profile
        if not resolved_risk_path and config_profile in {Profile.SPOT, Profile.DEV, Profile.DEMO}:
            if DEFAULT_SPOT_RISK_PATH.exists():
                resolved_risk_path = str(DEFAULT_SPOT_RISK_PATH)
                logger.info(
                    "Loading spot risk profile",
                    path=resolved_risk_path,
                    operation="risk_config",
                    stage="load",
                )

        path_obj: Path | None = Path(resolved_risk_path) if resolved_risk_path else None
        try:
            if path_obj and path_obj.exists():
                risk_config = self._risk_config_cls.from_json(str(path_obj))
            else:
                risk_config = self._risk_config_cls.from_env()
        except Exception:
            logger.exception(
                "Failed to load risk config; using defaults",
                operation="risk_config",
                stage="load",
            )
            risk_config = self._risk_config_cls()

        try:
            if getattr(ctx.config, "max_leverage", None):
                risk_config.max_leverage = int(ctx.config.max_leverage)
        except Exception as exc:
            logger.warning(
                "Failed to apply max leverage override",
                error=str(exc),
                exc_info=True,
                operation="risk_config",
                stage="override",
            )
        try:
            risk_config.reduce_only_mode = bool(ctx.config.reduce_only_mode)
        except Exception as exc:
            logger.warning(
                "Failed to sync reduce-only override into risk config",
                error=str(exc),
                exc_info=True,
                operation="risk_config",
                stage="override",
            )

        risk_manager = self._risk_manager_cls(config=risk_config, event_store=ctx.event_store)
        broker = ctx.broker or ctx.registry.broker
        try:
            if broker is not None and hasattr(broker, "get_position_risk"):
                risk_manager.set_risk_info_provider(broker.get_position_risk)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning(
                "Failed to set broker risk info provider",
                error=str(exc),
                exc_info=True,
                operation="risk_config",
                stage="risk_provider",
            )

        risk_manager.set_state_listener(self.on_risk_state_change)
        if controller is not None:
            controller.sync_with_risk_manager(risk_manager)
            risk_manager.set_reduce_only_mode(controller.reduce_only_mode, reason="config_init")

        registry = ctx.registry.with_updates(risk_manager=risk_manager)
        return ctx.with_updates(risk_manager=risk_manager, registry=registry)

    @property
    def _risk_config_cls(self) -> type[RiskConfig]:
        return cast(type[RiskConfig], RiskConfig)

    @property
    def _risk_manager_cls(self) -> type[LiveRiskManager]:
        return cast(type[LiveRiskManager], LiveRiskManager)


__all__ = ["RuntimeCoordinatorRiskMixin"]
