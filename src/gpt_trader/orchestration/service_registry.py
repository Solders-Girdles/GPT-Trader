"""Legacy service registry composition root.

DEPRECATED: Prefer `ApplicationContainer` in `gpt_trader.app.container` for all new code.
`ServiceRegistry` exists only to support older call sites while the codebase migrates.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from gpt_trader.orchestration.configuration import BotConfig

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.core.protocols import BrokerProtocol
    from gpt_trader.features.live_trade.risk.protocols import RiskManagerProtocol
    from gpt_trader.monitoring.notifications.service import NotificationService
    from gpt_trader.orchestration.protocols import EventStoreProtocol, OrdersStoreProtocol


@dataclass(frozen=True)
class ServiceRegistry:
    """Registry for shared services.

    .. deprecated:: 2.0
        ServiceRegistry is a legacy composition root pattern.
        Use ``ApplicationContainer`` instead for new code.

        Migration::

            # Old pattern
            registry = ServiceRegistry(config)
            bot = create_trading_bot(registry)

            # New pattern
            container = ApplicationContainer(config)
            bot = container.create_bot()

        Removal planned for v3.0.
    """

    config: BotConfig
    event_store: EventStoreProtocol | None = None
    orders_store: OrdersStoreProtocol | None = None
    broker: BrokerProtocol | None = None
    risk_manager: RiskManagerProtocol | None = None
    notification_service: NotificationService | None = None
    market_data_service: Any = None
    product_catalog: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Emit deprecation warning on instantiation."""
        warnings.warn(
            "ServiceRegistry is deprecated. Use ApplicationContainer instead. "
            "See migration guide in class docstring.",
            DeprecationWarning,
            stacklevel=2,
        )

    def with_updates(self, **kwargs: Any) -> ServiceRegistry:
        """Return a new registry with updated values.

        Note: Returns concrete ServiceRegistry type for compatibility with
        frozen dataclass pattern. Satisfies ServiceRegistryProtocol semantically.
        """
        return replace(self, **kwargs)


def empty_registry(config: BotConfig) -> ServiceRegistry:
    """Create an empty ServiceRegistry.

    .. deprecated:: 2.0
        Use ``ApplicationContainer`` instead.
    """
    return ServiceRegistry(config)
