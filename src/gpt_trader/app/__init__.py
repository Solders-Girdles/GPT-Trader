"""
Application composition root package.

This package contains the dependency injection container and composition root
for the trading bot application.
"""

from .container import ApplicationContainer, create_application_container
from .health_server import (
    DEFAULT_HEALTH_PORT,
    HealthServer,
    HealthState,
    add_health_check,
    mark_live,
    mark_ready,
    start_health_server,
)

__all__ = [
    "ApplicationContainer",
    "create_application_container",
    # Health server exports
    "HealthServer",
    "HealthState",
    "start_health_server",
    "mark_ready",
    "mark_live",
    "add_health_check",
    "DEFAULT_HEALTH_PORT",
]
