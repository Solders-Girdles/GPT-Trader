"""Shared fixtures for ConfigController tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot_v2.orchestration.config_controller import ConfigController
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.runtime_settings import RuntimeSettings
from bot_v2.utilities.config import ConfigBaselinePayload


@pytest.fixture
def mock_config_manager():
    """Mock ConfigManager with controlled behavior."""
    manager = MagicMock()

    # Default config for tests
    config = MagicMock(spec=BotConfig)
    config.reduce_only_mode = False
    config.derivatives_enabled = True

    manager.get_config.return_value = config
    manager.refresh_if_changed.return_value = None  # No change by default
    manager.replace_config = MagicMock()

    return manager


@pytest.fixture
def mock_runtime_settings():
    """Mock RuntimeSettings for tests."""
    return MagicMock(spec=RuntimeSettings)


@pytest.fixture
def mock_baseline_payload():
    """Mock ConfigBaselinePayload for diff tests."""
    payload = MagicMock(spec=ConfigBaselinePayload)
    payload.diff.return_value = {"test.setting": "old -> new"}
    return payload


@pytest.fixture
def mock_risk_manager():
    """Mock LiveRiskManager for tests."""
    manager = MagicMock()
    manager.is_reduce_only_mode.return_value = False
    manager.set_reduce_only_mode = MagicMock()
    return manager


@pytest.fixture
def sample_bot_config():
    """Create sample BotConfig for tests."""
    config = MagicMock(spec=BotConfig)
    config.reduce_only_mode = False
    config.derivatives_enabled = True
    config.symbols = ["BTC-PERP", "ETH-PERP"]
    config.profile = "test"
    return config


@pytest.fixture
def config_controller(sample_bot_config, mock_config_manager):
    """Create ConfigController with mocked dependencies."""
    with pytest.MonkeyPatch().context() as m:
        m.setattr("bot_v2.orchestration.config_controller.ConfigManager.from_config",
                  lambda config, settings=None: mock_config_manager)

        controller = ConfigController(sample_bot_config)
        controller._manager = mock_config_manager  # Ensure we have the mock

        return controller


@pytest.fixture
def config_controller_with_settings(sample_bot_config, mock_runtime_settings):
    """Create ConfigController with runtime settings."""
    with pytest.MonkeyPatch().context() as m:
        mock_manager = MagicMock()
        m.setattr("bot_v2.orchestration.config_controller.ConfigManager.from_config",
                  lambda config, settings=None: mock_manager)

        controller = ConfigController(sample_bot_config, settings=mock_runtime_settings)
        controller._manager = mock_manager

        return controller
