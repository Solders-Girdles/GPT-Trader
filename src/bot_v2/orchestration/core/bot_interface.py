"""Bot runtime interface for dependency injection.

This protocol defines the minimal interface required by orchestration services
that depend on PerpsBot, allowing us to break circular dependencies while
maintaining type safety.

By depending on this protocol instead of the concrete PerpsBot class, services
can be imported without triggering circular import errors.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

try:
    from typing import runtime_checkable
except ImportError:  # Python <3.8
    from typing_extensions import runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.brokerages.core.interfaces import IBrokerage, Product
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.monitoring.metrics_server import MetricsServer
    from bot_v2.orchestration.config_controller import ConfigController
    from bot_v2.orchestration.configuration import BotConfig
    from bot_v2.orchestration.execution_coordinator import ExecutionCoordinator
    from bot_v2.orchestration.service_registry import ServiceRegistry
    from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator
    from bot_v2.persistence.orders_store import OrdersStore


@runtime_checkable
class IBotRuntime(Protocol):
    """Protocol defining the bot runtime interface for dependency injection.

    This protocol defines the minimal set of properties and methods that
    orchestration services require from PerpsBot. By depending on this
    protocol instead of the concrete PerpsBot class, we break circular
    dependencies while maintaining type safety.

    Services should type hint with IBotRuntime instead of PerpsBot:

    Example:
        ```python
        from bot_v2.orchestration.core.bot_interface import IBotRuntime

        class MyService:
            def __init__(self, bot: IBotRuntime):
                self.bot = bot
                # Now can access bot.config, bot.broker, etc.
        ```

    This protocol is marked with @runtime_checkable to allow isinstance()
    checks for runtime verification.
    """

    # Core configuration and state
    @property
    def config(self) -> BotConfig:
        """Bot configuration."""
        ...

    @property
    def running(self) -> bool:
        """Whether the bot is currently running."""
        ...

    @running.setter
    def running(self, value: bool) -> None:
        """Set bot running state."""
        ...

    # Service references
    @property
    def broker(self) -> IBrokerage:
        """Brokerage interface for trading operations."""
        ...

    @broker.setter
    def broker(self, value: IBrokerage) -> None:
        """Set broker instance."""
        ...

    @property
    def registry(self) -> ServiceRegistry:
        """Service registry for dependency injection."""
        ...

    @registry.setter
    def registry(self, value: ServiceRegistry) -> None:
        """Set service registry."""
        ...

    @property
    def risk_manager(self) -> LiveRiskManager:
        """Risk management service."""
        ...

    @property
    def metrics_server(self) -> MetricsServer:
        """Metrics collection and export server."""
        ...

    @property
    def strategy_orchestrator(self) -> StrategyOrchestrator:
        """Strategy execution orchestrator."""
        ...

    @property
    def execution_coordinator(self) -> ExecutionCoordinator:
        """Order execution coordinator."""
        ...

    @property
    def config_controller(self) -> ConfigController:
        """Runtime configuration controller."""
        ...

    @property
    def orders_store(self) -> OrdersStore:
        """Order tracking and persistence store."""
        ...

    # State tracking
    @property
    def last_decisions(self) -> dict[str, Any]:
        """Most recent strategy decisions per symbol."""
        ...

    @property
    def mark_windows(self) -> dict[str, list[Decimal]]:
        """Recent mark prices per symbol."""
        ...

    @property
    def _product_map(self) -> dict[str, Product]:
        """Cached product specifications."""
        ...

    @property
    def _symbol_strategies(self) -> dict[str, Any]:
        """Per-symbol strategy instances."""
        ...

    @_symbol_strategies.setter
    def _symbol_strategies(self, value: dict[str, Any]) -> None:
        """Set symbol strategies."""
        ...

    @property
    def strategy(self) -> Any | None:
        """Default strategy instance."""
        ...

    @strategy.setter
    def strategy(self, value: Any) -> None:
        """Set default strategy."""
        ...

    # Core methods
    async def run_cycle(self) -> None:
        """Execute a single trading cycle."""
        ...

    def get_product(self, symbol: str) -> Product:
        """Get product specification for symbol."""
        ...

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: Any,
    ) -> None:
        """Execute a trading decision."""
        ...

    def stop(self) -> None:
        """Stop the bot gracefully."""
        ...
