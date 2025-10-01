"""Service registry scaffolding for orchestration dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import guards for type checkers only
    from bot_v2.data_providers import DataProvider
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.orchestration.configuration import BotConfig
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
    data_provider: DataProvider | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **overrides: Any) -> ServiceRegistry:
        """Return a new registry with selected fields replaced."""

        data = dict(self.__dict__)
        data.update(overrides)
        return replace(self, **data)


def empty_registry(config: BotConfig) -> ServiceRegistry:
    """Create a registry placeholder for a given configuration.

    This helper gives callers a consistent object to extend while the
    decomposition work progresses. It keeps the construction site explicit
    without introducing side effects during this refactor step.
    """

    return ServiceRegistry(config=config)
