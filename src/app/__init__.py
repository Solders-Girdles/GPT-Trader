"""
Application composition root package.

This package contains the dependency injection container and composition root
for the trading bot application.
"""

from .container import ApplicationContainer, create_application_container

__all__ = [
    "ApplicationContainer",
    "create_application_container",
]
