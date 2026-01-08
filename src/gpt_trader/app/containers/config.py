"""Config sub-container for ApplicationContainer.

This container manages configuration-related dependencies:
- Config controller (runtime config access)
- Profile loader (trading profile YAML loading)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpt_trader.app.config import BotConfig
from gpt_trader.orchestration.config_controller import ConfigController

if TYPE_CHECKING:
    from gpt_trader.app.config.profile_loader import ProfileLoader


class ConfigContainer:
    """Container for configuration-related dependencies.

    This container lazily initializes config controller and profile loader.

    Args:
        config: Bot configuration.
    """

    def __init__(self, config: BotConfig):
        self._config = config

        self._config_controller: ConfigController | None = None
        self._profile_loader: ProfileLoader | None = None

    @property
    def config_controller(self) -> ConfigController:
        """Get or create the config controller."""
        if self._config_controller is None:
            self._config_controller = ConfigController(self._config)
        return self._config_controller

    @property
    def profile_loader(self) -> ProfileLoader:
        """Get or create the profile loader.

        The profile loader handles loading and validating trading profile
        configurations from YAML files.
        """
        if self._profile_loader is None:
            from gpt_trader.app.config.profile_loader import ProfileLoader as PL

            self._profile_loader = PL()
        return self._profile_loader

    def reset_config(self) -> None:
        """Reset the config controller, forcing re-creation on next access."""
        self._config_controller = None
