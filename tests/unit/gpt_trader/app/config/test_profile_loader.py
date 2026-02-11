"""Tests for profile configuration builders."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from gpt_trader.app.config.profile_loader import (
    ProfileLoader,
    ProfileSchema,
    get_env_defaults_for_profile,
    get_profile_registry_entry_by_name,
    is_dev_profile,
)
from gpt_trader.config.types import Profile


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

    def test_logs_payload_on_yaml_parse_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """YAML parse failures log a normalized payload."""
        mock_logger = MagicMock()
        monkeypatch.setattr("gpt_trader.app.config.profile_loader.logger", mock_logger)

        yaml_path = tmp_path / "dev.yaml"
        yaml_path.write_text("invalid: [")

        loader = ProfileLoader(profiles_dir=tmp_path)
        schema = loader.load(Profile.DEV)

        assert schema.profile_name == "dev"
        assert mock_logger.warning.call_count == 1
        logged_kwargs = mock_logger.warning.call_args.kwargs
        details = logged_kwargs.get("details", {})
        assert details["profile"] == "dev"
        assert details["category"] == "yaml_parse"
        assert details["severity"] == "error"
        reason = details["reason"]
        assert "while parsing" in reason
        assert "\n" not in reason
        assert "Defaults are provided" in details["remediation"]
        assert "\n" not in details["remediation"]

    def test_logs_payload_when_profile_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing profile YAML logs a normalized warning payload."""
        mock_logger = MagicMock()
        monkeypatch.setattr("gpt_trader.app.config.profile_loader.logger", mock_logger)

        loader = ProfileLoader(profiles_dir=tmp_path)
        schema = loader.load(Profile.DEV)

        assert schema.profile_name == "dev"
        assert mock_logger.warning.call_count == 1
        logged_kwargs = mock_logger.warning.call_args.kwargs
        details = logged_kwargs.get("details", {})
        assert details["category"] == "missing_file"
        assert details["severity"] == "warning"
        assert details["reason"].startswith("Profile YAML not found")


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
                "daily_loss_limit_pct": 0.1,
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
        assert schema.risk.daily_loss_limit_pct == 0.1
        assert schema.execution.time_in_force == "FOK"
        assert schema.session.trading_days == ["monday", "wednesday", "friday"]
        assert schema.monitoring.update_interval == 120

    def test_from_yaml_uses_trading_execution_when_top_level_execution_missing(self) -> None:
        """Nested trading.execution should backfill execution settings."""
        data = {
            "profile_name": "nested-execution",
            "trading": {
                "execution": {
                    "time_in_force": "IOC",
                    "dry_run": True,
                    "mock_broker": True,
                    "use_limit_orders": True,
                    "market_order_fallback": False,
                }
            },
        }

        schema = ProfileSchema.from_yaml(data, "nested-execution")

        assert schema.execution.time_in_force == "IOC"
        assert schema.execution.dry_run is True
        assert schema.execution.mock_broker is True
        assert schema.execution.use_limit_orders is True
        assert schema.execution.market_order_fallback is False


class TestProfileRegistryHelpers:
    """Tests for profile registry helper functions."""

    def test_get_env_defaults_for_unknown_profile_uses_prod_defaults(self) -> None:
        defaults = get_env_defaults_for_profile("unknown-profile")

        assert defaults["COINBASE_SANDBOX"] == ("0", True)
        assert defaults["COINBASE_API_MODE"] == ("advanced", True)

    @pytest.mark.parametrize(
        ("profile_name", "expected"),
        [
            ("dev", True),
            ("DEV", True),
            ("paper", True),
            ("test", True),
            ("prod", False),
            ("canary", False),
            ("not-real", False),
            (None, False),
        ],
    )
    def test_is_dev_profile(self, profile_name: str | None, expected: bool) -> None:
        assert is_dev_profile(profile_name) is expected

    def test_registry_lookup_is_case_insensitive(self) -> None:
        entry = get_profile_registry_entry_by_name("CaNaRy")

        assert entry is not None
        assert entry.profile is Profile.CANARY
        assert entry.preflight_default is True
