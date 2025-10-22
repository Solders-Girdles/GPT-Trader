"""Tests for ConfigController refresh and diff computation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bot_v2.orchestration.config_controller import ConfigChange
from bot_v2.orchestration.configuration import BotConfig


class TestRefreshAndDiff:
    """Test ConfigController refresh logic and diff computation."""

    def test_refresh_if_changed_returns_none_when_no_change(self, config_controller) -> None:
        """Test refresh_if_changed returns None when manager reports no changes."""
        config_controller._manager.refresh_if_changed.return_value = None

        result = config_controller.refresh_if_changed()

        assert result is None
        assert config_controller._pending_change is None
        config_controller._manager.refresh_if_changed.assert_called_once()

    def test_refresh_if_changed_creates_config_change_when_updated(
        self, config_controller, sample_bot_config
    ) -> None:
        """Test refresh_if_changed creates ConfigChange when config is updated."""
        # Setup
        previous_config = MagicMock(spec=BotConfig)
        previous_config.reduce_only_mode = False
        updated_config = MagicMock(spec=BotConfig)
        updated_config.reduce_only_mode = True

        config_controller._manager.get_config.return_value = previous_config
        config_controller._manager.refresh_if_changed.return_value = updated_config

        # Mock diff computation
        expected_diff = {"test.setting": "old -> new"}
        with patch.object(
            config_controller, "_summarize_diff", return_value=expected_diff
        ) as mock_diff:
            result = config_controller.refresh_if_changed()

        # Verify result and state
        assert isinstance(result, ConfigChange)
        assert result.updated == updated_config
        assert result.diff == expected_diff
        assert config_controller._pending_change == result
        assert config_controller._reduce_only_mode_state is True

        # Verify method calls
        config_controller._manager.refresh_if_changed.assert_called_once()
        mock_diff.assert_called_once_with(previous_config, updated_config)

    def test_refresh_if_changed_updates_reduce_only_cache(self, config_controller) -> None:
        """Test refresh_if_changed updates reduce-only mode cache from updated config."""
        previous_config = MagicMock(spec=BotConfig)
        previous_config.reduce_only_mode = False
        updated_config = MagicMock(spec=BotConfig)
        updated_config.reduce_only_mode = True

        config_controller._manager.get_config.return_value = previous_config
        config_controller._manager.refresh_if_changed.return_value = updated_config

        with patch.object(config_controller, "_summarize_diff", return_value={}):
            config_controller.refresh_if_changed()

        assert config_controller._reduce_only_mode_state is True

    def test_refresh_if_changed_handles_boolean_conversion(self, config_controller) -> None:
        """Test refresh_if_changed properly converts reduce_only_mode to boolean."""
        previous_config = MagicMock(spec=BotConfig)
        previous_config.reduce_only_mode = False
        updated_config = MagicMock(spec=BotConfig)
        updated_config.reduce_only_mode = "enabled"  # Truthy string

        config_controller._manager.get_config.return_value = previous_config
        config_controller._manager.refresh_if_changed.return_value = updated_config

        with patch.object(config_controller, "_summarize_diff", return_value={}):
            config_controller.refresh_if_changed()

        assert config_controller._reduce_only_mode_state is True

    def test_summarize_diff_creates_baseline_payloads(self, config_controller) -> None:
        """Test _summarize_diff creates ConfigBaselinePayload objects."""
        current_config = MagicMock(spec=BotConfig)
        current_config.derivatives_enabled = True
        updated_config = MagicMock(spec=BotConfig)
        updated_config.derivatives_enabled = False

        # Test that ConfigBaselinePayload.from_config is called with correct arguments
        with patch(
            "bot_v2.orchestration.config_controller.ConfigBaselinePayload.from_config"
        ) as mock_from_config:
            # Setup mock payloads
            current_payload = MagicMock()
            updated_payload = MagicMock()
            expected_diff = {"some.setting": "old -> new"}

            # Configure the diff method to return the expected dict
            current_payload.diff.return_value = expected_diff

            mock_from_config.side_effect = [current_payload, updated_payload]

            result = config_controller._summarize_diff(current_config, updated_config)

            # Verify payload creation with derivatives flag
            mock_from_config.assert_any_call(current_config, derivatives_enabled=True)
            mock_from_config.assert_any_call(updated_config, derivatives_enabled=False)
            assert result == expected_diff

    def test_summarize_diff_handles_missing_derivatives_enabled(self, config_controller) -> None:
        """Test _summarize_diff handles missing derivatives_enabled attribute."""
        current_config = MagicMock(spec=BotConfig)
        del current_config.derivatives_enabled  # Remove attribute
        updated_config = MagicMock(spec=BotConfig)

        with patch(
            "bot_v2.orchestration.config_controller.ConfigBaselinePayload.from_config"
        ) as mock_from_config:
            current_payload = MagicMock()
            updated_payload = MagicMock()
            expected_diff = {"test": "diff"}

            # Configure the diff method to return the expected dict
            current_payload.diff.return_value = expected_diff

            mock_from_config.side_effect = [current_payload, updated_payload]

            result = config_controller._summarize_diff(current_config, updated_config)

            # Should default derivatives_enabled to False when missing
            mock_from_config.assert_any_call(current_config, derivatives_enabled=False)
            assert result == expected_diff

    def test_summarize_diff_returns_diff_result(self, config_controller) -> None:
        """Test _summarize_diff returns the diff from payloads."""
        current_config = MagicMock(spec=BotConfig)
        updated_config = MagicMock(spec=BotConfig)

        with patch(
            "bot_v2.orchestration.config_controller.ConfigBaselinePayload.from_config"
        ) as mock_from_config:
            current_payload = MagicMock()
            updated_payload = MagicMock()
            expected_diff = {"risk.max_position": "0.5 -> 0.7"}

            # Configure the diff method to return the expected dict
            current_payload.diff.return_value = expected_diff

            mock_from_config.side_effect = [current_payload, updated_payload]

            result = config_controller._summarize_diff(current_config, updated_config)

            assert result == expected_diff
            current_payload.diff.assert_called_once_with(updated_payload)

    def test_refresh_if_changed_handles_current_property_exception(self, config_controller) -> None:
        """Test refresh_if_changed propagates exception when current property fails."""
        config_controller._manager.get_config.side_effect = RuntimeError("Config unavailable")

        with pytest.raises(RuntimeError, match="Config unavailable"):
            config_controller.refresh_if_changed()

    def test_refresh_if_changed_stores_pending_change_even_with_empty_diff(
        self, config_controller
    ) -> None:
        """Test refresh_if_changed stores pending change even when diff is empty."""
        previous_config = MagicMock(spec=BotConfig)
        updated_config = MagicMock(spec=BotConfig)
        updated_config.reduce_only_mode = True

        config_controller._manager.get_config.return_value = previous_config
        config_controller._manager.refresh_if_changed.return_value = updated_config

        with patch.object(config_controller, "_summarize_diff", return_value={}):
            result = config_controller.refresh_if_changed()

        # Should still create and store ConfigChange even with empty diff
        assert isinstance(result, ConfigChange)
        assert result.updated == updated_config
        assert result.diff == {}
        assert config_controller._pending_change == result

    def test_refresh_if_changed_multiple_calls_accumulate_state(self, config_controller) -> None:
        """Test multiple refresh_if_changed calls properly accumulate state."""
        # First refresh
        previous_config = MagicMock(spec=BotConfig)
        first_updated = MagicMock(spec=BotConfig)
        first_updated.reduce_only_mode = True

        config_controller._manager.get_config.return_value = previous_config
        config_controller._manager.refresh_if_changed.return_value = first_updated

        with patch.object(config_controller, "_summarize_diff", return_value={"first": "change"}):
            first_result = config_controller.refresh_if_changed()

        # Second refresh
        second_updated = MagicMock(spec=BotConfig)
        second_updated.reduce_only_mode = False

        config_controller._manager.get_config.return_value = first_updated
        config_controller._manager.refresh_if_changed.return_value = second_updated

        with patch.object(config_controller, "_summarize_diff", return_value={"second": "change"}):
            second_result = config_controller.refresh_if_changed()

        # Verify each refresh stored its own change
        assert first_result.diff == {"first": "change"}
        assert second_result.diff == {"second": "change"}
        assert config_controller._pending_change == second_result
        assert config_controller._reduce_only_mode_state is False

    def test_refresh_if_changed_preserves_pending_change_until_consumed(
        self, config_controller
    ) -> None:
        """Test refresh_if_changed preserves pending change until consumed."""
        updated_config = MagicMock(spec=BotConfig)
        updated_config.reduce_only_mode = True
        config_controller._manager.get_config.return_value = MagicMock(spec=BotConfig)
        config_controller._manager.refresh_if_changed.return_value = updated_config

        with patch.object(config_controller, "_summarize_diff", return_value={"test": "diff"}):
            refresh_result = config_controller.refresh_if_changed()

        # Should be the same instance
        consume_result = config_controller.consume_pending_change()
        assert refresh_result is consume_result

        # After consumption, should be None
        assert config_controller.consume_pending_change() is None
