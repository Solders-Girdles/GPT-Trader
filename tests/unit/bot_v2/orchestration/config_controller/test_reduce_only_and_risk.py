"""Tests for ConfigController reduce-only mode and risk manager synchronization."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestReduceOnlyAndRisk:
    """Test ConfigController reduce-only mode management and risk manager integration."""

    def test_sync_with_risk_manager_when_risk_manager_enforces_reduce_only(self, config_controller, mock_risk_manager) -> None:
        """Test sync_with_risk_manager updates config when risk manager enforces reduce-only."""
        # Setup: risk manager says reduce-only, config doesn't
        config = MagicMock()
        config.reduce_only_mode = False
        config.with_overrides.return_value = MagicMock(reduce_only_mode=True)

        config_controller._manager.get_config.return_value = config
        mock_risk_manager.is_reduce_only_mode.return_value = True

        config_controller.sync_with_risk_manager(mock_risk_manager)

        # Verify config was updated
        config.with_overrides.assert_called_once_with(reduce_only_mode=True)
        config_controller._manager.replace_config.assert_called_once()
        # Verify cache was updated
        assert config_controller._reduce_only_mode_state is True

    def test_sync_with_risk_manager_when_config_already_reduce_only(self, config_controller, mock_risk_manager) -> None:
        """Test sync_with_risk_manager doesn't update when config already matches risk manager."""
        config = MagicMock()
        config.reduce_only_mode = True

        config_controller._manager.get_config.return_value = config
        mock_risk_manager.is_reduce_only_mode.return_value = True

        config_controller.sync_with_risk_manager(mock_risk_manager)

        # Should not update config
        config.with_overrides.assert_not_called()
        config_controller._manager.replace_config.assert_not_called()
        # Cache should still be updated to match
        assert config_controller._reduce_only_mode_state is True

    def test_sync_with_risk_manager_when_both_not_reduce_only(self, config_controller, mock_risk_manager) -> None:
        """Test sync_with_risk_manager when neither config nor risk manager require reduce-only."""
        config = MagicMock()
        config.reduce_only_mode = False

        config_controller._manager.get_config.return_value = config
        mock_risk_manager.is_reduce_only_mode.return_value = False

        config_controller.sync_with_risk_manager(mock_risk_manager)

        # Should not update config
        config.with_overrides.assert_not_called()
        config_controller._manager.replace_config.assert_not_called()
        # Cache should be updated
        assert config_controller._reduce_only_mode_state is False

    def test_sync_with_risk_manager_updates_cache_directly_when_no_config_change(self, config_controller, mock_risk_manager) -> None:
        """Test sync_with_risk_manager updates cache directly when config already matches."""
        config = MagicMock()
        config.reduce_only_mode = True

        config_controller._manager.get_config.return_value = config
        mock_risk_manager.is_reduce_only_mode.return_value = True

        config_controller.sync_with_risk_manager(mock_risk_manager)

        # Cache should be updated even though config wasn't changed
        assert config_controller._reduce_only_mode_state is True

    def test_set_reduce_only_mode_returns_false_when_no_change(self, config_controller, mock_risk_manager) -> None:
        """Test set_reduce_only_mode returns False when mode is already set."""
        config_controller._reduce_only_mode_state = True

        result = config_controller.set_reduce_only_mode(True, reason="test", risk_manager=mock_risk_manager)

        assert result is False
        mock_risk_manager.set_reduce_only_mode.assert_not_called()

    def test_set_reduce_only_mode_enables_with_risk_manager(self, config_controller, mock_risk_manager) -> None:
        """Test set_reduce_only_mode enables mode and updates risk manager."""
        config = MagicMock()
        config.reduce_only_mode = False
        config.with_overrides.return_value = MagicMock(reduce_only_mode=True)

        config_controller._manager.get_config.return_value = config
        config_controller._reduce_only_mode_state = False

        result = config_controller.set_reduce_only_mode(True, reason="test reason", risk_manager=mock_risk_manager)

        # Verify returns True for change
        assert result is True
        # Verify config updated
        config.with_overrides.assert_called_once_with(reduce_only_mode=True)
        config_controller._manager.replace_config.assert_called_once()
        # Verify risk manager updated
        mock_risk_manager.set_reduce_only_mode.assert_called_once_with(True, reason="test reason")
        # Verify cache updated
        assert config_controller._reduce_only_mode_state is True

    def test_set_reduce_only_mode_disables_with_risk_manager(self, config_controller, mock_risk_manager) -> None:
        """Test set_reduce_only_mode disables mode and updates risk manager."""
        config = MagicMock()
        config.reduce_only_mode = True
        config.with_overrides.return_value = MagicMock(reduce_only_mode=False)

        config_controller._manager.get_config.return_value = config
        config_controller._reduce_only_mode_state = True

        result = config_controller.set_reduce_only_mode(False, reason="disable reason", risk_manager=mock_risk_manager)

        assert result is True
        config.with_overrides.assert_called_once_with(reduce_only_mode=False)
        mock_risk_manager.set_reduce_only_mode.assert_called_once_with(False, reason="disable reason")
        assert config_controller._reduce_only_mode_state is False

    def test_set_reduce_only_mode_without_risk_manager(self, config_controller) -> None:
        """Test set_reduce_only_mode works without risk manager."""
        config = MagicMock()
        config.reduce_only_mode = False
        config.with_overrides.return_value = MagicMock(reduce_only_mode=True)

        config_controller._manager.get_config.return_value = config
        config_controller._reduce_only_mode_state = False

        result = config_controller.set_reduce_only_mode(True, reason="test")

        assert result is True
        config.with_overrides.assert_called_once_with(reduce_only_mode=True)
        config_controller._manager.replace_config.assert_called_once()
        assert config_controller._reduce_only_mode_state is True

    def test_is_reduce_only_mode_returns_true_when_risk_manager_enforces(self, config_controller, mock_risk_manager) -> None:
        """Test is_reduce_only_mode returns True when risk manager enforces reduce-only."""
        config_controller._reduce_only_mode_state = False
        mock_risk_manager.is_reduce_only_mode.return_value = True

        result = config_controller.is_reduce_only_mode(mock_risk_manager)

        assert result is True

    def test_is_reduce_only_mode_returns_cached_when_risk_manager_allows(self, config_controller, mock_risk_manager) -> None:
        """Test is_reduce_only_mode returns cached state when risk manager allows."""
        config_controller._reduce_only_mode_state = True
        mock_risk_manager.is_reduce_only_mode.return_value = False

        result = config_controller.is_reduce_only_mode(mock_risk_manager)

        assert result is True

    def test_is_reduce_only_mode_without_risk_manager(self, config_controller) -> None:
        """Test is_reduce_only_mode returns cached state when no risk manager provided."""
        config_controller._reduce_only_mode_state = True

        result = config_controller.is_reduce_only_mode(None)

        assert result is True

    def test_is_reduce_only_mode_with_none_risk_manager(self, config_controller) -> None:
        """Test is_reduce_only_mode handles None risk manager gracefully."""
        config_controller._reduce_only_mode_state = False

        result = config_controller.is_reduce_only_mode(None)

        assert result is False

    def test_apply_risk_update_returns_false_when_no_change(self, config_controller) -> None:
        """Test apply_risk_update returns False when mode is already set."""
        config_controller._reduce_only_mode_state = True

        result = config_controller.apply_risk_update(True)

        assert result is False

    def test_apply_risk_update_enables_reduce_only(self, config_controller) -> None:
        """Test apply_risk_update enables reduce-only mode."""
        config = MagicMock()
        config.reduce_only_mode = False
        config.with_overrides.return_value = MagicMock(reduce_only_mode=True)

        config_controller._manager.get_config.return_value = config
        config_controller._reduce_only_mode_state = False

        result = config_controller.apply_risk_update(True)

        assert result is True
        config.with_overrides.assert_called_once_with(reduce_only_mode=True)
        config_controller._manager.replace_config.assert_called_once()
        assert config_controller._reduce_only_mode_state is True

    def test_apply_risk_update_disables_reduce_only(self, config_controller) -> None:
        """Test apply_risk_update disables reduce-only mode."""
        config = MagicMock()
        config.reduce_only_mode = True
        config.with_overrides.return_value = MagicMock(reduce_only_mode=False)

        config_controller._manager.get_config.return_value = config
        config_controller._reduce_only_mode_state = True

        result = config_controller.apply_risk_update(False)

        assert result is True
        config.with_overrides.assert_called_once_with(reduce_only_mode=False)
        config_controller._manager.replace_config.assert_called_once()
        assert config_controller._reduce_only_mode_state is False

    def test_set_reduce_only_mode_idempotent_when_already_set(self, config_controller) -> None:
        """Test set_reduce_only_mode is idempotent when mode is already set."""
        config_controller._reduce_only_mode_state = True

        result = config_controller.set_reduce_only_mode(True, reason="test")

        assert result is False  # No change occurred
        config_controller._manager.get_config.return_value.with_overrides.assert_not_called()

    def test_reduce_only_mode_precedence_with_risk_manager(self, config_controller, mock_risk_manager) -> None:
        """Test that risk manager takes precedence over cached state."""
        # Cache says not reduce-only, risk manager says reduce-only
        config_controller._reduce_only_mode_state = False
        mock_risk_manager.is_reduce_only_mode.return_value = True

        result = config_controller.is_reduce_only_mode(mock_risk_manager)

        assert result is True  # Risk manager should win

    def test_reduce_only_mode_caching_behavior(self, config_controller, mock_risk_manager) -> None:
        """Test that is_reduce_only_mode properly caches risk manager state."""
        # First call should check risk manager
        mock_risk_manager.is_reduce_only_mode.return_value = True
        result1 = config_controller.is_reduce_only_mode(mock_risk_manager)
        assert result1 is True

        # Cache should be unaffected by is_reduce_only_mode
        assert config_controller._reduce_only_mode_state is False

        # Subsequent call without risk manager should use cache
        result2 = config_controller.is_reduce_only_mode(None)
        assert result2 is False  # Should use cached state, not risk manager state
