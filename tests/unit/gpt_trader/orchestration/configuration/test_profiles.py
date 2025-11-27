"""Tests for profile configuration builders."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.config.types import Profile
from gpt_trader.orchestration.configuration.profiles import (
    build_canary_config,
    build_production_config,
    build_profile_config,
)


def _create_mock_config_factory() -> MagicMock:
    """Create a mock config factory that returns a mock BotConfig."""

    def factory(**kwargs):
        config = MagicMock()
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config

    return MagicMock(side_effect=factory)


class TestBuildProfileConfig:
    """Tests for build_profile_config function."""

    def test_canary_profile_delegates_to_build_canary(self) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(Profile.CANARY, factory)

        # Canary profile should be set
        assert result.profile == Profile.CANARY

    def test_dev_profile_sets_mock_broker(self) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(Profile.DEV, factory)

        assert result.mock_broker is True
        assert result.mock_fills is True
        assert result.dry_run is True
        assert result.max_position_size == Decimal("10000")

    def test_demo_profile_conservative_settings(self) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(Profile.DEMO, factory)

        assert result.max_position_size == Decimal("100")
        assert result.max_leverage == 1
        assert result.enable_shorts is False

    def test_spot_profile_settings(self) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(Profile.SPOT, factory)

        assert result.max_position_size == Decimal("50000")
        assert result.max_leverage == 1
        assert result.enable_shorts is False
        assert result.mock_broker is False
        assert result.mock_fills is False

    def test_production_profile_uses_build_production(self) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(Profile.PROD, factory)

        assert result.max_position_size == Decimal("50000")
        assert result.max_leverage == 3
        assert result.enable_shorts is True


class TestBuildCanaryConfig:
    """Tests for build_canary_config function."""

    def test_uses_default_values_when_no_yaml(self) -> None:
        factory = _create_mock_config_factory()

        with patch(
            "gpt_trader.orchestration.configuration.profiles.path_registry"
        ) as mock_registry:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_registry.PROJECT_ROOT = MagicMock()
            mock_registry.PROJECT_ROOT.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = (
                mock_path
            )

            result = build_canary_config(Profile.CANARY, factory)

        # Check default values
        assert result.profile == Profile.CANARY
        assert result.symbols == ["BTC-USD"]
        assert result.reduce_only_mode is True
        assert result.max_leverage == 1
        assert result.max_position_size == Decimal("500")

    def test_time_in_force_defaults_to_ioc(self) -> None:
        factory = _create_mock_config_factory()

        with patch(
            "gpt_trader.orchestration.configuration.profiles.path_registry"
        ) as mock_registry:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_registry.PROJECT_ROOT = MagicMock()
            mock_registry.PROJECT_ROOT.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = (
                mock_path
            )

            result = build_canary_config(Profile.CANARY, factory)

        assert result.time_in_force == "IOC"


class TestBuildProductionConfig:
    """Tests for build_production_config function."""

    def test_production_settings(self) -> None:
        factory = _create_mock_config_factory()

        result = build_production_config(Profile.PROD, factory)

        assert result.profile == Profile.PROD
        assert result.max_position_size == Decimal("50000")
        assert result.max_leverage == 3
        assert result.enable_shorts is True

    def test_uses_provided_profile(self) -> None:
        factory = _create_mock_config_factory()

        # Can be called with any profile that defaults to production settings
        result = build_production_config(Profile.TEST, factory)

        assert result.profile == Profile.TEST


class TestProfileMapping:
    """Tests for profile to config mapping."""

    @pytest.mark.parametrize(
        "profile",
        [Profile.DEV, Profile.DEMO, Profile.SPOT, Profile.CANARY, Profile.PROD],
    )
    def test_all_profiles_return_config(self, profile: Profile) -> None:
        factory = _create_mock_config_factory()

        with patch(
            "gpt_trader.orchestration.configuration.profiles.path_registry"
        ) as mock_registry:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_registry.PROJECT_ROOT = MagicMock()
            mock_registry.PROJECT_ROOT.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = (
                mock_path
            )

            result = build_profile_config(profile, factory)

        assert result is not None
        assert result.profile == profile
