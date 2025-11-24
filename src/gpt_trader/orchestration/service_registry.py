from dataclasses import dataclass, replace, field
from typing import Any, Optional, Dict
from gpt_trader.orchestration.configuration import BotConfig

@dataclass(frozen=True)
class ServiceRegistry:
    """
    Registry for shared services.
    Legacy composition root pattern.
    """
    config: BotConfig
    event_store: Any = None
    orders_store: Any = None
    broker: Any = None
    market_data_service: Any = None
    product_catalog: Any = None
    runtime_settings: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **kwargs) -> 'ServiceRegistry':
        return replace(self, **kwargs)

def empty_registry(config: BotConfig) -> ServiceRegistry:
    return ServiceRegistry(config)
