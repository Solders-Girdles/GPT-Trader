"""Tests for profile configuration builders."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from gpt_trader.config.types import Profile
from gpt_trader.orchestration.configuration.profile_loader import (
    ProfileLoader,
    ProfileSchema,
    load_profile,
)
from gpt_trader.orchestration.configuration.profiles import (
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


class TestProfileLoader:
    """Tests for ProfileLoader class."""

    def test_load_from_yaml_file(self) -> None:
        """Test loading profile from YAML file."""
        with TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir)
            yaml_content = """
profile_name: "test"
environment: "test"
trading:
  symbols:
    - "BTC-USD"
  mode: "normal"
  interval: 30
risk_management:
  max_leverage: 2
  max_position_size: 5000
  position_fraction: 0.05
  enable_shorts: true
execution:
  time_in_force: "IOC"
  dry_run: true
  mock_broker: true
monitoring:
  log_level: "DEBUG"
"""
            yaml_path = profiles_dir / "test.yaml"
            yaml_path.write_text(yaml_content)

            loader = ProfileLoader(profiles_dir=profiles_dir)
            schema = loader.load(Profile.TEST)

            assert schema.profile_name == "test"
            assert schema.trading.symbols == ["BTC-USD"]
            assert schema.trading.interval == 30
            assert schema.risk.max_leverage == 2
            assert schema.risk.max_position_size == Decimal("5000")
            assert schema.risk.enable_shorts is True
            assert schema.execution.time_in_force == "IOC"
            assert schema.execution.dry_run is True
            assert schema.monitoring.log_level == "DEBUG"

    def test_fallback_to_hardcoded_defaults(self) -> None:
        """Test fallback to hardcoded defaults when YAML doesn't exist."""
        with TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir)  # Empty directory, no YAML files
            loader = ProfileLoader(profiles_dir=profiles_dir)

            schema = loader.load(Profile.DEV)

            # Should use hardcoded defaults for DEV profile
            assert schema.profile_name == "dev"
            assert schema.execution.mock_broker is True
            assert schema.execution.dry_run is True

    def test_hardcoded_defaults_exist_for_all_profiles(self) -> None:
        """Test that hardcoded defaults exist for all Profile enum values."""
        with TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir)  # Empty directory
            loader = ProfileLoader(profiles_dir=profiles_dir)

            for profile in Profile:
                schema = loader.load(profile)
                assert schema is not None
                assert schema.profile_name == profile.value


class TestProfileSchema:
    """Tests for ProfileSchema dataclass."""

    def test_from_yaml_minimal(self) -> None:
        """Test creating ProfileSchema from minimal YAML data."""
        data = {
            "profile_name": "minimal",
        }
        schema = ProfileSchema.from_yaml(data, "minimal")

        assert schema.profile_name == "minimal"
        # Should have defaults for everything else
        assert schema.trading.symbols == ["BTC-USD"]
        assert schema.risk.max_leverage == 1

    def test_from_yaml_full(self) -> None:
        """Test creating ProfileSchema from complete YAML data."""
        data = {
            "profile_name": "full",
            "environment": "production",
            "description": "Full test profile",
            "trading": {
                "symbols": ["ETH-USD", "BTC-USD"],
                "mode": "reduce_only",
                "interval": 120,
            },
            "strategy": {
                "type": "mean_reversion",
                "short_ma_period": 10,
                "long_ma_period": 50,
            },
            "risk_management": {
                "max_leverage": 5,
                "max_position_size": 100000,
                "position_fraction": 0.2,
                "enable_shorts": True,
                "daily_loss_limit": 1000,
            },
            "execution": {
                "time_in_force": "FOK",
                "dry_run": False,
                "mock_broker": False,
            },
            "session": {
                "start_time": "09:00",
                "end_time": "17:00",
                "trading_days": ["monday", "wednesday", "friday"],
            },
            "monitoring": {
                "log_level": "WARNING",
                "metrics": {"interval_seconds": 120},
                "status_enabled": False,
            },
        }
        schema = ProfileSchema.from_yaml(data, "full")

        assert schema.profile_name == "full"
        assert schema.environment == "production"
        assert schema.trading.symbols == ["ETH-USD", "BTC-USD"]
        assert schema.trading.mode == "reduce_only"
        assert schema.strategy.type == "mean_reversion"
        assert schema.strategy.short_ma_period == 10
        assert schema.risk.max_leverage == 5
        assert schema.risk.daily_loss_limit == Decimal("1000")
        assert schema.execution.time_in_force == "FOK"
        assert schema.session.trading_days == ["monday", "wednesday", "friday"]
        assert schema.monitoring.update_interval == 120


class TestBuildProfileConfig:
    """Tests for build_profile_config function."""

    def test_dev_profile_sets_mock_broker(self) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(Profile.DEV, factory)

        assert result.mock_broker is True
        assert result.dry_run is True

    def test_demo_profile_conservative_settings(self) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(Profile.DEMO, factory)

        # Demo should be conservative
        assert result.enable_shorts is False

    def test_production_profile_settings(self) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(Profile.PROD, factory)

        # Production should have higher limits and shorts enabled
        assert result.enable_shorts is True

    def test_canary_profile_reduce_only(self) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(Profile.CANARY, factory)

        # Canary should be in reduce_only mode
        assert result.reduce_only_mode is True

    def test_spot_profile_no_shorts(self) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(Profile.SPOT, factory)

        # Spot should have no shorts
        assert result.enable_shorts is False
        assert result.mock_broker is False


class TestProfileMapping:
    """Tests for profile to config mapping."""

    @pytest.mark.parametrize(
        "profile",
        [Profile.DEV, Profile.DEMO, Profile.SPOT, Profile.CANARY, Profile.PROD, Profile.TEST],
    )
    def test_all_profiles_return_config(self, profile: Profile) -> None:
        factory = _create_mock_config_factory()

        result = build_profile_config(profile, factory)

        assert result is not None
        assert result.profile == profile


class TestLoadProfile:
    """Tests for load_profile convenience function."""

    def test_load_profile_returns_schema(self) -> None:
        """Test that load_profile returns a valid ProfileSchema."""
        schema = load_profile(Profile.DEV)

        assert isinstance(schema, ProfileSchema)
        assert schema.profile_name == "dev"

    @pytest.mark.parametrize("profile", list(Profile))
    def test_load_all_profiles(self, profile: Profile) -> None:
        """Test that all profiles can be loaded."""
        schema = load_profile(profile)

        assert isinstance(schema, ProfileSchema)
        assert schema.profile_name == profile.value
