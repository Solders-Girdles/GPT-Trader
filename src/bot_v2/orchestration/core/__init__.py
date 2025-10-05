"""Core orchestration interfaces and protocols.

This module provides runtime interfaces used throughout the orchestration layer
to break circular dependencies and enable dependency injection.
"""

from bot_v2.orchestration.core.bot_interface import IBotRuntime

__all__ = ["IBotRuntime"]
