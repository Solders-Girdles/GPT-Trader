"""Tests for bot_config state module."""

from __future__ import annotations

from gpt_trader.orchestration.configuration.bot_config.state import ConfigState


class TestConfigStateInit:
    """Tests for ConfigState initialization."""

    def test_runtime_settings_is_none(self) -> None:
        state = ConfigState()
        assert state.runtime_settings is None

    def test_profile_value_is_none(self) -> None:
        state = ConfigState()
        assert state.profile_value is None

    def test_overrides_snapshot_is_empty_dict(self) -> None:
        state = ConfigState()
        assert state.overrides_snapshot == {}

    def test_config_snapshot_is_none(self) -> None:
        state = ConfigState()
        assert state.config_snapshot is None


class TestConfigStateMutability:
    """Tests for ConfigState mutable attributes."""

    def test_can_set_profile_value(self) -> None:
        state = ConfigState()
        state.profile_value = "aggressive"
        assert state.profile_value == "aggressive"

    def test_can_update_overrides_snapshot(self) -> None:
        state = ConfigState()
        state.overrides_snapshot = {"key": "value"}
        assert state.overrides_snapshot == {"key": "value"}

    def test_can_set_config_snapshot(self) -> None:
        state = ConfigState()
        state.config_snapshot = {"symbols": ["BTC-USD"]}
        assert state.config_snapshot == {"symbols": ["BTC-USD"]}

    def test_multiple_states_are_independent(self) -> None:
        state1 = ConfigState()
        state2 = ConfigState()

        state1.profile_value = "aggressive"
        state2.profile_value = "conservative"

        assert state1.profile_value == "aggressive"
        assert state2.profile_value == "conservative"
