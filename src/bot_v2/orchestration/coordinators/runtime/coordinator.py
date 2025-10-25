"""Composite runtime coordinator assembled from focused mixins."""

from __future__ import annotations

from typing import Any

from ..base import BaseCoordinator, CoordinatorContext
from .broker import RuntimeCoordinatorBrokerMixin
from .models import BrokerBootstrapArtifacts, BrokerBootstrapError
from .reconcile import RuntimeCoordinatorReconcileMixin
from .reduce_only import RuntimeCoordinatorReduceOnlyMixin
from .risk import RuntimeCoordinatorRiskMixin


class RuntimeCoordinator(
    RuntimeCoordinatorReconcileMixin,
    RuntimeCoordinatorReduceOnlyMixin,
    RuntimeCoordinatorRiskMixin,
    RuntimeCoordinatorBrokerMixin,
    BaseCoordinator,
):
    """Handles broker and risk bootstrapping plus runtime safety toggles."""

    def __init__(
        self,
        context: CoordinatorContext,
        *,
        config_controller: Any | None = None,
        strategy_orchestrator: Any | None = None,
        execution_coordinator: Any | None = None,
        product_cache: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(context)
        self._config_controller = config_controller or context.config_controller
        self._strategy_orchestrator = strategy_orchestrator or context.strategy_orchestrator
        self._execution_coordinator = execution_coordinator or context.execution_coordinator
        self._product_cache = product_cache or context.product_cache
        self._reduce_only_mode = False

    @property
    def name(self) -> str:
        return "runtime"

    def update_context(self, context: CoordinatorContext) -> None:
        super().update_context(context)
        if context.config_controller is not None:
            self._config_controller = context.config_controller
        if context.strategy_orchestrator is not None:
            self._strategy_orchestrator = context.strategy_orchestrator
        if context.execution_coordinator is not None:
            self._execution_coordinator = context.execution_coordinator
        if context.product_cache is not None:
            self._product_cache = context.product_cache

    def initialize(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        ctx = context or self.context

        broker_ctx = self._init_broker(ctx)
        if isinstance(broker_ctx, CoordinatorContext):
            ctx = broker_ctx
        self.update_context(ctx)
        ctx = self.context

        risk_ctx = self._init_risk_manager(ctx)
        if isinstance(risk_ctx, CoordinatorContext):
            ctx = risk_ctx
        self.update_context(ctx)

        return self.context

    def bootstrap(self) -> None:
        """Compatibility wrapper to align with legacy API."""

        self.initialize(self.context)


__all__ = ["RuntimeCoordinator", "BrokerBootstrapArtifacts", "BrokerBootstrapError"]
