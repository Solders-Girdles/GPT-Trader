"""Tests for ConfigController state management and initialization."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bot_v2.orchestration.config_controller import ConfigChange, ConfigController
from bot_v2.orchestration.configuration import BotConfig


class TestStateManagement:
    """Test ConfigController initialization, state access, and basic operations."""

    def test_initialization_stores_manager_and_sets_reduce_only_mode(
        self, sample_bot_config, mock_config_manager
    ) -> None:
        """Test ConfigController stores manager and initializes reduce-only mode from config."""
        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "bot_v2.orchestration.config_controller.ConfigManager.from_config",
                lambda config, settings=None: mock_config_manager,
            )

            controller = ConfigController(sample_bot_config)

            assert controller._manager == mock_config_manager
            assert controller._reduce_only_mode_state == sample_bot_config.reduce_only_mode
            assert controller._pending_change is None

    def test_initialization_with_runtime_settings(
        self, sample_bot_config, mock_runtime_settings
    ) -> None:
        """Test ConfigController passes settings to ConfigManager."""
        with pytest.MonkeyPatch().context() as m:
            mock_manager = MagicMock()
            mock_from_config = MagicMock(return_value=mock_manager)
            m.setattr(
                "bot_v2.orchestration.config_controller.ConfigManager.from_config", mock_from_config
            )

            ConfigController(sample_bot_config, settings=mock_runtime_settings)

            mock_from_config.assert_called_once_with(
                sample_bot_config, settings=mock_runtime_settings
            )

    def test_current_property_returns_config_from_manager(
        self, config_controller, sample_bot_config
    ) -> None:
        """Test current property returns config from manager."""
        config_controller._manager.get_config.return_value = sample_bot_config

        result = config_controller.current

        assert result == sample_bot_config
        config_controller._manager.get_config.assert_called_once()

    def test_current_property_raises_when_manager_not_initialized(self, config_controller) -> None:
        """Test current property raises RuntimeError when manager returns None."""
        config_controller._manager.get_config.return_value = None

        with pytest.raises(RuntimeError, match="Config manager not initialized"):
            _ = config_controller.current

    def test_reduce_only_mode_property_returns_cached_state(self, config_controller) -> None:
        """Test reduce_only_mode property returns cached state."""
        config_controller._reduce_only_mode_state = True

        result = config_controller.reduce_only_mode

        assert result is True

    def test_reduce_only_mode_property_returns_boolean(self, config_controller) -> None:
        """Test reduce_only_mode property always returns boolean."""
        config_controller._reduce_only_mode_state = "truthy"
        result = config_controller.reduce_only_mode
        assert result is True

        config_controller._reduce_only_mode_state = 0
        result = config_controller.reduce_only_mode
        assert result is False

    def test_set_current_config_updates_manager_and_cache(
        self, config_controller, sample_bot_config
    ) -> None:
        """Test _set_current_config updates manager and cached reduce_only mode."""
        new_config = MagicMock(spec=BotConfig)
        new_config.reduce_only_mode = True

        config_controller._set_current_config(new_config)

        config_controller._manager.replace_config.assert_called_once_with(new_config)
        assert config_controller._reduce_only_mode_state is True

    def test_set_current_config_converts_reduce_only_to_boolean(
        self, config_controller, sample_bot_config
    ) -> None:
        """Test _set_current_config converts reduce_only_mode to boolean."""
        new_config = MagicMock(spec=BotConfig)
        new_config.reduce_only_mode = "enabled"

        config_controller._set_current_config(new_config)

        assert config_controller._reduce_only_mode_state is True

        new_config.reduce_only_mode = []
        config_controller._set_current_config(new_config)
        assert config_controller._reduce_only_mode_state is False

    def test_consume_pending_change_returns_and_clears_pending_change(
        self, config_controller
    ) -> None:
        """Test consume_pending_change returns pending change and clears it."""
        mock_change = MagicMock(spec=ConfigChange)
        config_controller._pending_change = mock_change

        result = config_controller.consume_pending_change()

        assert result == mock_change
        assert config_controller._pending_change is None

    def test_consume_pending_change_returns_none_when_no_pending_change(
        self, config_controller
    ) -> None:
        """Test consume_pending_change returns None when no pending change."""
        config_controller._pending_change = None

        result = config_controller.consume_pending_change()

        assert result is None

    def test_consume_pending_change_clears_state_even_if_change_is_none(
        self, config_controller
    ) -> None:
        """Test consume_pending_change clears state even when change is None."""
        config_controller._pending_change = None

        result = config_controller.consume_pending_change()

        assert result is None
        assert config_controller._pending_change is None

    def test_consume_pending_change_idempotent(self, config_controller) -> None:
        """Test consume_pending_change is idempotent."""
        mock_change = MagicMock(spec=ConfigChange)
        config_controller._pending_change = mock_change

        # First call should return the change
        first_result = config_controller.consume_pending_change()
        assert first_result == mock_change

        # Second call should return None
        second_result = config_controller.consume_pending_change()
        assert second_result is None

    def test_state_isolation_between_instances(self, sample_bot_config) -> None:
        """Test that different ConfigController instances have isolated state."""
        with pytest.MonkeyPatch().context() as m:
            mock_manager1 = MagicMock()
            mock_manager2 = MagicMock()
            m.setattr(
                "bot_v2.orchestration.config_controller.ConfigManager.from_config",
                MagicMock(side_effect=[mock_manager1, mock_manager2]),
            )

            config1 = sample_bot_config
            config2 = MagicMock(spec=BotConfig)
            config2.reduce_only_mode = True

            controller1 = ConfigController(config1)
            controller2 = ConfigController(config2)

            # Verify different managers
            assert controller1._manager == mock_manager1
            assert controller2._manager == mock_manager2

            # Verify different reduce-only states
            assert controller1._reduce_only_mode_state == config1.reduce_only_mode
            assert controller2._reduce_only_mode_state == config2.reduce_only_mode

            # Verify pending changes are isolated
            controller1._pending_change = MagicMock()
            assert controller2._pending_change is None
