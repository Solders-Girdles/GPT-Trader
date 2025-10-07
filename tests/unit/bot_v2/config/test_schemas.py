"""Tests for configuration validation schemas."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from bot_v2.config.schemas import BotConfigSchema, ConfigValidationResult
from bot_v2.orchestration.configuration import Profile


class TestBotConfigSchema:
    """Test the pydantic schema for BotConfig validation."""

    def test_valid_config(self):
        """Test that a valid configuration passes validation."""
        config_data = {
            "profile": Profile.DEV,
            "dry_run": True,
            "symbols": ["BTC-USD", "ETH-USD"],
            "derivatives_enabled": False,
            "update_interval": 5,
            "short_ma": 5,
            "long_ma": 20,
            "target_leverage": 2,
            "trailing_stop_pct": 0.01,
            "enable_shorts": False,
            "max_position_size": Decimal("1000"),
            "max_leverage": 3,
            "reduce_only_mode": False,
            "mock_broker": True,
            "mock_fills": True,
            "enable_order_preview": False,
            "account_telemetry_interval": 300,
            "daily_loss_limit": Decimal("100"),
            "time_in_force": "GTC",
            "perps_enable_streaming": False,
            "perps_stream_level": 1,
            "perps_paper_trading": False,
            "perps_force_mock": False,
            "perps_position_fraction": None,
            "perps_skip_startup_reconcile": False,
        }

        schema = BotConfigSchema(**config_data)
        assert schema.profile == Profile.DEV
        assert schema.dry_run is True
        assert schema.symbols == ["BTC-USD", "ETH-USD"]

    def test_invalid_max_leverage(self):
        """Test validation of max_leverage field."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "max_leverage": 0,  # Invalid: must be positive
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "max_leverage_too_small" in str(exc_info.value)

    def test_invalid_max_leverage_type(self):
        """Test validation of max_leverage type."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "max_leverage": "invalid",  # Invalid: must be integer
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "max_leverage_invalid" in str(exc_info.value)

    def test_invalid_update_interval(self):
        """Test validation of update_interval."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "update_interval": -1,  # Invalid: must be positive
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "update_interval_too_small" in str(exc_info.value)

    def test_invalid_max_position_size(self):
        """Test validation of max_position_size."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "max_position_size": Decimal("0"),  # Invalid: must be positive
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "max_position_size_too_small" in str(exc_info.value)

    def test_invalid_symbols_empty(self):
        """Test validation of empty symbols list."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": [],  # Invalid: cannot be empty
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "symbols_empty" in str(exc_info.value)

    def test_invalid_symbols_invalid_values(self):
        """Test validation of symbols with invalid values."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD", "", "ETH-USD"],  # Invalid: empty string
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "symbols_invalid_values" in str(exc_info.value)

    def test_invalid_time_in_force(self):
        """Test validation of time_in_force."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "time_in_force": "INVALID",  # Invalid: not supported
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "time_in_force_unsupported" in str(exc_info.value)

    def test_profile_canary_constraints(self):
        """Test profile-specific constraints for canary profile."""
        config_data = {
            "profile": Profile.CANARY,
            "symbols": ["BTC-USD"],
            "reduce_only_mode": False,  # Invalid: must be True for canary
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "canary_reduce_only_required" in str(exc_info.value)

    def test_profile_canary_max_leverage_constraint(self):
        """Test canary profile max_leverage constraint."""
        config_data = {
            "profile": Profile.CANARY,
            "symbols": ["BTC-USD"],
            "reduce_only_mode": True,
            "max_leverage": 2,  # Invalid: must be 1 for canary
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "canary_max_leverage_too_high" in str(exc_info.value)

    def test_profile_canary_time_in_force_constraint(self):
        """Test canary profile time_in_force constraint."""
        config_data = {
            "profile": Profile.CANARY,
            "symbols": ["BTC-USD"],
            "reduce_only_mode": True,
            "time_in_force": "GTC",  # Invalid: must be IOC for canary
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "canary_time_in_force_invalid" in str(exc_info.value)

    def test_profile_spot_constraints(self):
        """Test profile-specific constraints for spot profile."""
        config_data = {
            "profile": Profile.SPOT,
            "symbols": ["BTC-USD"],
            "enable_shorts": True,  # Invalid: must be False for spot
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "spot_shorts_not_allowed" in str(exc_info.value)

    def test_invalid_ma_periods(self):
        """Test validation that short_ma must be less than long_ma."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "short_ma": 20,
            "long_ma": 5,  # Invalid: short_ma >= long_ma
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "ma_periods_invalid" in str(exc_info.value)

    def test_invalid_perps_position_fraction_range(self):
        """Test validation of perps_position_fraction range."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "perps_position_fraction": 1.5,  # Invalid: must be <= 1.0
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "perps_position_fraction_invalid_range" in str(exc_info.value)

    def test_invalid_perps_stream_level(self):
        """Test validation of perps_stream_level."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "perps_stream_level": 0,  # Invalid: must be >= 1
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "perps_stream_level_too_small" in str(exc_info.value)

    def test_negative_trailing_stop_pct(self):
        """Test validation of trailing_stop_pct."""
        config_data = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "trailing_stop_pct": -0.01,  # Invalid: must be non-negative
        }

        with pytest.raises(ValidationError) as exc_info:
            BotConfigSchema(**config_data)

        assert "trailing_stop_pct_negative" in str(exc_info.value)


class TestConfigValidationResult:
    """Test the ConfigValidationResult model."""

    def test_valid_result(self):
        """Test a valid validation result."""
        result = ConfigValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.has_errors is False
        assert result.has_warnings is False

    def test_invalid_result(self):
        """Test an invalid validation result."""
        result = ConfigValidationResult(
            is_valid=False, errors=["error1", "error2"], warnings=["warning1"]
        )
        assert result.is_valid is False
        assert result.errors == ["error1", "error2"]
        assert result.warnings == ["warning1"]
        assert result.has_errors is True
        assert result.has_warnings is True
