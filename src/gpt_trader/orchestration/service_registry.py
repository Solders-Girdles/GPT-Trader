from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from gpt_trader.orchestration.configuration import BotConfig

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.core.protocols import BrokerProtocol
    from gpt_trader.features.live_trade.risk.protocols import RiskManagerProtocol
    from gpt_trader.orchestration.protocols import EventStoreProtocol, OrdersStoreProtocol


@dataclass(frozen=True)
class ServiceRegistry:
    """
    Registry for shared services.
    Legacy composition root pattern.
    """

    config: BotConfig
    event_store: EventStoreProtocol | None = None
    orders_store: OrdersStoreProtocol | None = None
    broker: BrokerProtocol | None = None
    risk_manager: RiskManagerProtocol | None = None
    market_data_service: Any = None
    product_catalog: Any = None
    runtime_settings: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **kwargs: Any) -> ServiceRegistry:
        return replace(self, **kwargs)


def empty_registry(config: BotConfig) -> ServiceRegistry:
    return ServiceRegistry(config)
