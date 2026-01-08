"""Sub-containers for ApplicationContainer.

This package contains focused containers that manage related groups of
dependencies. The main ApplicationContainer delegates to these sub-containers
to keep concerns separated and the codebase maintainable.
"""

from gpt_trader.app.containers.brokerage import BrokerageContainer
from gpt_trader.app.containers.config import ConfigContainer
from gpt_trader.app.containers.observability import ObservabilityContainer
from gpt_trader.app.containers.persistence import PersistenceContainer
from gpt_trader.app.containers.risk_validation import RiskValidationContainer

__all__ = [
    "BrokerageContainer",
    "ConfigContainer",
    "ObservabilityContainer",
    "PersistenceContainer",
    "RiskValidationContainer",
]
