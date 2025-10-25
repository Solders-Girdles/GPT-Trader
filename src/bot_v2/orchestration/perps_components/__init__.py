"""PerpsBot components package for focused functionality.

This package contains specialized components that were previously
consolidated in the large perps_bot.py file. The separation provides:

- Focused responsibility for each component
- Improved testability and maintainability
- Clear separation of concerns
- Better error handling and logging

Modules:
- lifecycle_management: Bot lifecycle and session coordination
- session_coordination: Trading session management and validation
- component_management: Component lifecycle and health monitoring
- configuration_guardian: Configuration monitoring and drift detection
"""

from .component_management import ComponentManagementService
from .configuration_guardian import ConfigurationGuardianService
from .lifecycle_management import PerpsBotLifecycleManager
from .session_coordination import SessionCoordinationService

__all__ = [
    "PerpsBotLifecycleManager",
    "SessionCoordinationService",
    "ComponentManagementService",
    "ConfigurationGuardianService",
]
