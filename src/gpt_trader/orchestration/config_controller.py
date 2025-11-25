"""
Configuration controller for managing bot configuration state.
"""

from typing import Generic, TypeVar

from gpt_trader.orchestration.configuration import BotConfig

T = TypeVar("T", bound=BotConfig)


class ConfigController(Generic[T]):
    """
    Manages the active configuration for the bot.
    Provides a stable interface for accessing the current config
    even if it changes during runtime (reloading).
    """

    def __init__(self, initial_config: T):
        self._current = initial_config

    @property
    def current(self) -> T:
        """Get the current active configuration."""
        return self._current

    def update(self, new_config: T) -> None:
        """Update the current configuration."""
        self._current = new_config
