"""Service registry scaffolding for orchestration dependencies.

.. deprecated::
    ServiceRegistry is deprecated for construction duties. Use the composition root
    pattern with bot_v2.app.ApplicationContainer instead. ServiceRegistry remains
    available for runtime dependency access in existing code, but new code should
    use the container for service construction and dependency injection.

    Migration guide:
    - Replace ServiceRegistry construction with ApplicationContainer
    - Use container.create_service_registry() for backward compatibility
    - See bot_v2.app.container for the modern dependency injection approach
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:  # pragma: no cover - import guards for type checkers only
    from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
    from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.orchestration.configuration import BotConfig
    from bot_v2.orchestration.runtime_settings import RuntimeSettings
    from bot_v2.orchestration.state_manager import ReduceOnlyModeStateManager
    from bot_v2.persistence.event_store import EventStore
    from bot_v2.persistence.orders_store import OrdersStore


@dataclass(frozen=True)
class ServiceRegistry:
    """Container for orchestration-level dependencies.

    The registry keeps orchestration construction explicit by bundling the
    objects the trading loop depends on (event store, brokerage, risk manager,
    etc.). Future work will populate this via a dedicated bootstrapper so the
    core bot can accept a ready-to-run bundle instead of instantiating
    components ad-hoc inside its constructor.
    """

    config: BotConfig
    event_store: EventStore | None = None
    orders_store: OrdersStore | None = None
    risk_manager: LiveRiskManager | None = None
    broker: IBrokerage | None = None
    market_data_service: MarketDataService | None = None
    product_catalog: ProductCatalog | None = None
    runtime_settings: RuntimeSettings | None = None
    reduce_only_state_manager: ReduceOnlyModeStateManager | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **overrides: Any) -> ServiceRegistry:
        """Return a new registry with selected fields replaced."""

        data = dict(self.__dict__)
        extras = data.get("extras")
        if isinstance(extras, dict):
            data["extras"] = dict(extras)
        data.update(overrides)
        if "extras" in data and isinstance(data["extras"], dict):
            data["extras"] = dict(data["extras"])
        return replace(self, **data)

    def get_intx_portfolio_service(self) -> Any:
        return cast(Any, self.extras.get("intx_portfolio_service"))


def empty_registry(config: BotConfig) -> ServiceRegistry:
    """Create a registry placeholder for a given configuration.

    This helper gives callers a consistent object to extend while the
    decomposition work progresses. It keeps the construction site explicit
    without introducing side effects during this refactor step.
    """

    return ServiceRegistry(config=config)
