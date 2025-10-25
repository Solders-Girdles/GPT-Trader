"""Manager package for standardized management interfaces.

This package provides base classes and standardized interfaces for all manager
types in the GPT-Trader orchestration system. It ensures consistent behavior,
error handling, logging, and lifecycle management across all manager implementations.

Manager Types:
- BaseManager: Abstract base for all managers
- StatefulManager: Base for managers that maintain state
- ConfigurableManager: Base for managers that need configuration
"""

from .base import (
    BaseManager,
    ConfigurableManager,
    ManagerInterface,
    ManagerStatus,
    StatefulManager,
    create_manager,
)

__all__ = [
    "BaseManager",
    "ConfigurableManager",
    "ManagerInterface",
    "ManagerStatus",
    "StatefulManager",
    "create_manager",
]
