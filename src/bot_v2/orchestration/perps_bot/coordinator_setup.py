"""Coordinator wiring helpers."""

from __future__ import annotations

from bot_v2.orchestration.coordinators import (
    CoordinatorContext,
    CoordinatorRegistry,
    ExecutionCoordinator,
    RuntimeCoordinator,
    StrategyCoordinator,
    TelemetryCoordinator,
)
from bot_v2.orchestration.system_monitor import SystemMonitor


class PerpsBotCoordinatorMixin:
    """Encapsulates coordinator stack configuration."""

    def _setup_coordinator_stack(self) -> None:
        """Instantiate coordinator context, registry, and wiring."""

        self._coordinator_context = CoordinatorContext(
            config=self.config,
            registry=self.registry,
            event_store=self.event_store,
            orders_store=self.orders_store,
            symbols=tuple(self.symbols),
            bot_id=self.bot_id,
            runtime_state=self._state,
            config_controller=self.config_controller,
            strategy_orchestrator=self.strategy_orchestrator,
            execution_coordinator=None,
            strategy_coordinator=None,
            session_guard=self._session_guard,
            configuration_guardian=self.configuration_guardian,
            system_monitor=None,
            set_reduce_only_mode=self.set_reduce_only_mode,
            shutdown_hook=self.shutdown,
            set_running_flag=lambda value: setattr(self, "running", value),
        )

        self._coordinator_registry = CoordinatorRegistry(self._coordinator_context)
        self._register_coordinators()

        self.system_monitor = SystemMonitor(self)
        self._coordinator_context = self._coordinator_context.with_updates(
            system_monitor=self.system_monitor,
            execution_coordinator=self.execution_coordinator,
            strategy_coordinator=self.strategy_coordinator,
        )
        self._coordinator_registry._context = self._coordinator_context  # type: ignore[attr-defined]

        for coordinator in (
            self.runtime_coordinator,
            self.execution_coordinator,
            self.strategy_coordinator,
            self.telemetry_coordinator,
        ):
            if hasattr(coordinator, "update_context"):
                coordinator.update_context(self._coordinator_context)

    def _register_coordinators(self) -> None:
        """Register orchestrator coordinators in dependency order."""

        runtime = RuntimeCoordinator(
            self._coordinator_context,
            config_controller=self.config_controller,
            strategy_orchestrator=self.strategy_orchestrator,
            execution_coordinator=None,
            product_cache=self._state.product_map,
        )
        execution = ExecutionCoordinator(self._coordinator_context)
        strategy = StrategyCoordinator(self._coordinator_context)
        telemetry = TelemetryCoordinator(self._coordinator_context)

        self._coordinator_context = self._coordinator_context.with_updates(
            execution_coordinator=execution,
            strategy_coordinator=strategy,
        )
        self._coordinator_registry._context = self._coordinator_context  # type: ignore[attr-defined]

        for coordinator in (runtime, execution, strategy, telemetry):
            coordinator.update_context(self._coordinator_context)

        self._coordinator_registry.register(runtime)
        self._coordinator_registry.register(execution)
        self._coordinator_registry.register(strategy)
        self._coordinator_registry.register(telemetry)

        # Inject the centralized state manager into the risk manager if available
        if (
            hasattr(self.registry, "reduce_only_state_manager")
            and self.registry.reduce_only_state_manager is not None
        ):
            if hasattr(self.registry.risk_manager, "_centralized_state_manager"):
                self.registry.risk_manager._centralized_state_manager = (
                    self.registry.reduce_only_state_manager
                )
