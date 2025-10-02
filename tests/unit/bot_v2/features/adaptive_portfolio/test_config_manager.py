"""
Comprehensive tests for adaptive portfolio ConfigManager.

Covers file loading, validation, hot-reload, tier logic, and backup.
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from bot_v2.features.adaptive_portfolio.config_manager import (
    ConfigManager,
    get_config_manager,
    get_current_tier,
    load_portfolio_config,
    validate_portfolio_config,
)
from bot_v2.features.adaptive_portfolio.types import (
    CostStructure,
    MarketConstraints,
    PortfolioConfig,
    PositionConstraints,
    RiskProfile,
    TierConfig,
    TradingRules,
)


@pytest.fixture
def valid_config_dict():
    """Valid configuration as dictionary."""
    return {
        "version": "1.0",
        "last_updated": "2025-01-01",
        "description": "Test config",
        "tiers": {
            "micro": {
                "name": "Micro",
                "range": [500, 2500],
                "positions": {"min": 2, "max": 3, "target": 2},
                "min_position_size": 150,
                "strategies": ["momentum"],
                "risk": {
                    "daily_limit_pct": 1.0,
                    "quarterly_limit_pct": 8.0,
                    "position_stop_loss_pct": 5.0,
                    "max_sector_exposure_pct": 50.0,
                },
                "trading": {
                    "max_trades_per_week": 3,
                    "account_type": "cash",
                    "settlement_days": 2,
                    "pdt_compliant": True,
                },
            },
            "small": {
                "name": "Small",
                "range": [2500, 10000],
                "positions": {"min": 3, "max": 5, "target": 4},
                "min_position_size": 500,
                "strategies": ["momentum", "mean_reversion"],
                "risk": {
                    "daily_limit_pct": 1.5,
                    "quarterly_limit_pct": 12.0,
                    "position_stop_loss_pct": 6.0,
                    "max_sector_exposure_pct": 60.0,
                },
                "trading": {
                    "max_trades_per_week": 6,
                    "account_type": "cash",
                    "settlement_days": 2,
                    "pdt_compliant": True,
                },
            },
        },
        "costs": {
            "commission_per_trade": 0.0,
            "spread_estimate_pct": 0.05,
            "slippage_pct": 0.1,
            "financing_rate_annual_pct": 8.0,
        },
        "market_constraints": {
            "min_share_price": 5.0,
            "max_share_price": 1000.0,
            "min_daily_volume": 100000,
            "excluded_sectors": ["crypto"],
            "excluded_symbols": ["PENNY"],
            "market_hours_only": True,
        },
        "validation": {
            "min_account_size": 500,
            "max_positions_any_tier": 20,
        },
        "rebalancing": {
            "frequency_days": 7,
            "threshold_pct": 10.0,
        },
    }


@pytest.fixture
def temp_config_file(tmp_path, valid_config_dict):
    """Create temporary config file."""
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(valid_config_dict, indent=2))
    return config_file


class TestConfigManagerFileOperations:
    """Test file loading and error handling."""

    def test_load_missing_file(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        manager = ConfigManager(tmp_path / "nonexistent.json")
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            manager.load_config()

    def test_load_invalid_json(self, tmp_path):
        """Should raise error for malformed JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ invalid json }")

        manager = ConfigManager(bad_file)
        with pytest.raises(json.JSONDecodeError):
            manager.load_config()

    def test_load_valid_config(self, temp_config_file):
        """Should successfully load valid config."""
        manager = ConfigManager(temp_config_file)
        config = manager.load_config()

        assert isinstance(config, PortfolioConfig)
        assert config.version == "1.0"
        assert len(config.tiers) == 2
        assert "micro" in config.tiers
        assert "small" in config.tiers

    def test_hot_reload_caching(self, temp_config_file):
        """Should cache config and only reload when file changes."""
        manager = ConfigManager(temp_config_file)

        # First load
        config1 = manager.load_config()
        assert config1 is not None

        # Second load without changes - should return cached
        config2 = manager.load_config()
        assert config2 is config1  # Same object

    def test_hot_reload_detects_changes(self, temp_config_file, valid_config_dict):
        """Should detect file modification and reload."""
        manager = ConfigManager(temp_config_file)

        # Initial load
        config1 = manager.load_config()
        original_version = config1.version

        # Modify file
        time.sleep(0.01)  # Ensure mtime changes
        valid_config_dict["version"] = "2.0"
        temp_config_file.write_text(json.dumps(valid_config_dict, indent=2))

        # Should detect change
        config2 = manager.load_config()
        assert config2.version == "2.0"
        assert config2.version != original_version

    def test_force_reload(self, temp_config_file):
        """Should reload even if file unchanged when force_reload=True."""
        manager = ConfigManager(temp_config_file)

        config1 = manager.load_config()
        config2 = manager.load_config(force_reload=True)

        # Different objects even though file unchanged
        assert config1 is not config2
        assert config1.version == config2.version


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_validate_position_sizing_math_error(self, temp_config_file, valid_config_dict):
        """Should detect when min_capital/max_positions < min_position_size."""
        # Make micro tier invalid: 500 / 10 = 50, but min_position = 150
        valid_config_dict["tiers"]["micro"]["positions"]["max"] = 10
        temp_config_file.write_text(json.dumps(valid_config_dict))

        manager = ConfigManager(temp_config_file)
        with pytest.raises(ValueError, match="Invalid configuration"):
            manager.load_config()

    def test_validate_excessive_daily_risk(self, temp_config_file, valid_config_dict):
        """Should reject daily risk limit > 10%."""
        valid_config_dict["tiers"]["micro"]["risk"]["daily_limit_pct"] = 15.0
        temp_config_file.write_text(json.dumps(valid_config_dict))

        manager = ConfigManager(temp_config_file)
        with pytest.raises(ValueError, match="Daily risk limit too high"):
            manager.load_config()

    def test_validate_excessive_quarterly_risk(self, temp_config_file, valid_config_dict):
        """Should reject quarterly risk limit > 50%."""
        valid_config_dict["tiers"]["micro"]["risk"]["quarterly_limit_pct"] = 60.0
        temp_config_file.write_text(json.dumps(valid_config_dict))

        manager = ConfigManager(temp_config_file)
        with pytest.raises(ValueError, match="Quarterly risk limit too high"):
            manager.load_config()

    def test_validate_overlapping_tier_ranges(self, temp_config_file, valid_config_dict):
        """Should detect overlapping tier ranges."""
        # Make ranges overlap: micro [500, 3000], small [2500, 10000]
        valid_config_dict["tiers"]["micro"]["range"] = [500, 3000]
        temp_config_file.write_text(json.dumps(valid_config_dict))

        manager = ConfigManager(temp_config_file)
        with pytest.raises(ValueError, match="Tier ranges overlap"):
            manager.load_config()

    def test_validate_pdt_warning(self, temp_config_file, valid_config_dict):
        """Should warn about PDT compliance for accounts < 25k."""
        # Make small tier non-PDT compliant (max is 10k, should warn)
        valid_config_dict["tiers"]["small"]["trading"]["pdt_compliant"] = False
        temp_config_file.write_text(json.dumps(valid_config_dict))

        manager = ConfigManager(temp_config_file)
        config = manager.load_config()
        result = manager.validate_config(config)

        assert result.is_valid
        assert any("PDT" in w for w in result.warnings)

    def test_validate_high_spread_warning(self, temp_config_file, valid_config_dict):
        """Should warn about high spread estimates."""
        valid_config_dict["costs"]["spread_estimate_pct"] = 2.0
        temp_config_file.write_text(json.dumps(valid_config_dict))

        manager = ConfigManager(temp_config_file)
        config = manager.load_config()
        result = manager.validate_config(config)

        assert result.is_valid
        assert any("Spread estimate seems high" in w for w in result.warnings)

    def test_validate_high_slippage_warning(self, temp_config_file, valid_config_dict):
        """Should warn about high slippage estimates."""
        valid_config_dict["costs"]["slippage_pct"] = 1.0
        temp_config_file.write_text(json.dumps(valid_config_dict))

        manager = ConfigManager(temp_config_file)
        config = manager.load_config()
        result = manager.validate_config(config)

        assert result.is_valid
        assert any("Slippage estimate seems high" in w for w in result.warnings)

    def test_validate_too_many_positions_suggestion(self, temp_config_file, valid_config_dict):
        """Should suggest fewer positions for small portfolios."""
        # Micro tier with many positions but small capital
        # Math: 2000 / 12 = 166.67 > 150 (valid), but triggers suggestion
        # Suggestion triggers when: max_positions > 10 AND min_capital < 10000
        valid_config_dict["tiers"]["micro"]["positions"]["max"] = 12
        valid_config_dict["tiers"]["micro"]["min_position_size"] = 150
        valid_config_dict["tiers"]["micro"]["range"] = [2000, 2500]  # Changed to avoid division error
        temp_config_file.write_text(json.dumps(valid_config_dict))

        manager = ConfigManager(temp_config_file)
        config = manager.load_config()
        result = manager.validate_config(config)

        assert any("fewer positions" in s.lower() for s in result.suggestions)


class TestTierLogic:
    """Test tier selection and range handling."""

    def test_get_tier_for_capital_micro(self, temp_config_file):
        """Should select micro tier for capital in range."""
        manager = ConfigManager(temp_config_file)
        manager.load_config()

        assert manager.get_tier_for_capital(500) == "micro"
        assert manager.get_tier_for_capital(1000) == "micro"
        assert manager.get_tier_for_capital(2499) == "micro"

    def test_get_tier_for_capital_small(self, temp_config_file):
        """Should select small tier for capital in range."""
        manager = ConfigManager(temp_config_file)
        manager.load_config()

        assert manager.get_tier_for_capital(2500) == "small"
        assert manager.get_tier_for_capital(5000) == "small"
        assert manager.get_tier_for_capital(9999) == "small"

    def test_get_tier_for_capital_above_all_ranges(self, temp_config_file):
        """Should default to 'large' tier for capital above all ranges."""
        manager = ConfigManager(temp_config_file)
        manager.load_config()

        # Config only has micro and small, so 50k should default to "large"
        assert manager.get_tier_for_capital(50000) == "large"


class TestSaveAndBackup:
    """Test configuration persistence and backup."""

    def test_save_config_without_backup(self, temp_config_file):
        """Should save config without creating backup."""
        manager = ConfigManager(temp_config_file)
        config = manager.load_config()

        # Modify and save without backup
        config.version = "2.0"
        manager.save_config(config, backup=False)

        # Reload and verify
        manager2 = ConfigManager(temp_config_file)
        reloaded = manager2.load_config()
        assert reloaded.version == "2.0"

        # No backup file should exist
        backup_path = temp_config_file.with_suffix(".json.backup")
        assert not backup_path.exists()

    def test_save_config_with_backup(self, temp_config_file):
        """Should create backup before saving."""
        manager = ConfigManager(temp_config_file)
        config = manager.load_config()

        # Save with backup
        config.version = "3.0"
        manager.save_config(config, backup=True)

        # Check backup exists
        backup_path = temp_config_file.with_suffix(".json.backup")
        assert backup_path.exists()

        # Backup should contain original version
        with open(backup_path) as f:
            backup_data = json.load(f)
        assert backup_data["version"] == "1.0"

        # Main file should have new version
        with open(temp_config_file) as f:
            current_data = json.load(f)
        assert current_data["version"] == "3.0"

    def test_save_updates_cache(self, temp_config_file):
        """Should update internal cache after save."""
        manager = ConfigManager(temp_config_file)
        config = manager.load_config()

        config.version = "4.0"
        manager.save_config(config, backup=False)

        # Cached version should match without reload
        assert manager._config.version == "4.0"
        assert manager._last_modified is not None

    def test_round_trip_serialization(self, temp_config_file):
        """Config should survive round-trip through save/load."""
        manager = ConfigManager(temp_config_file)
        original = manager.load_config()

        # Save and reload
        manager.save_config(original, backup=False)
        manager2 = ConfigManager(temp_config_file)
        reloaded = manager2.load_config()

        # Should be equivalent
        assert reloaded.version == original.version
        assert len(reloaded.tiers) == len(original.tiers)
        assert reloaded.costs.commission_per_trade == original.costs.commission_per_trade


class TestGlobalHelpers:
    """Test module-level convenience functions."""

    def test_get_config_manager_singleton(self, temp_config_file):
        """Should return singleton instance."""
        # Reset global state
        import bot_v2.features.adaptive_portfolio.config_manager as cm

        cm._config_manager = None

        manager1 = get_config_manager(str(temp_config_file))
        manager2 = get_config_manager()

        assert manager1 is manager2

    def test_get_config_manager_new_path_creates_new_instance(self, tmp_path, valid_config_dict):
        """Providing new path should create new instance."""
        import bot_v2.features.adaptive_portfolio.config_manager as cm

        cm._config_manager = None

        file1 = tmp_path / "config1.json"
        file1.write_text(json.dumps(valid_config_dict))

        file2 = tmp_path / "config2.json"
        valid_config_dict["version"] = "2.0"
        file2.write_text(json.dumps(valid_config_dict))

        manager1 = get_config_manager(str(file1))
        manager2 = get_config_manager(str(file2))

        assert manager1 is not manager2
        assert manager2.config_path == file2

    def test_load_portfolio_config_convenience(self, temp_config_file):
        """Should load config via convenience function."""
        import bot_v2.features.adaptive_portfolio.config_manager as cm

        cm._config_manager = None

        config = load_portfolio_config(str(temp_config_file))
        assert isinstance(config, PortfolioConfig)
        assert config.version == "1.0"

    def test_validate_portfolio_config_convenience(self, temp_config_file):
        """Should validate config via convenience function."""
        import bot_v2.features.adaptive_portfolio.config_manager as cm

        cm._config_manager = None

        result = validate_portfolio_config(str(temp_config_file))
        assert result.is_valid

    def test_get_current_tier_convenience(self, temp_config_file):
        """Should get tier via convenience function."""
        import bot_v2.features.adaptive_portfolio.config_manager as cm

        cm._config_manager = None

        tier = get_current_tier(1000, str(temp_config_file))
        assert tier == "micro"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_tiers(self, tmp_path, valid_config_dict):
        """Should handle config with no tiers."""
        valid_config_dict["tiers"] = {}
        config_file = tmp_path / "empty.json"
        config_file.write_text(json.dumps(valid_config_dict))

        manager = ConfigManager(config_file)
        config = manager.load_config()

        # Validation should still work
        result = manager.validate_config(config)
        assert result.is_valid

    def test_missing_optional_fields(self, tmp_path):
        """Should handle configs with minimal required fields."""
        minimal_config = {
            "version": "1.0",
            "last_updated": "2025-01-01",
            "description": "Minimal",
            "tiers": {},
            "costs": {
                "commission_per_trade": 0,
                "spread_estimate_pct": 0.05,
                "slippage_pct": 0.1,
                "financing_rate_annual_pct": 8.0,
            },
            "market_constraints": {
                "min_share_price": 5.0,
                "max_share_price": 1000.0,
                "min_daily_volume": 100000,
                "excluded_sectors": [],
                "excluded_symbols": [],
                "market_hours_only": True,
            },
            "validation": {},
            "rebalancing": {},
        }

        config_file = tmp_path / "minimal.json"
        config_file.write_text(json.dumps(minimal_config))

        manager = ConfigManager(config_file)
        config = manager.load_config()
        assert config.version == "1.0"

    def test_tier_boundary_exact_match(self, temp_config_file):
        """Should handle exact boundary matches correctly."""
        manager = ConfigManager(temp_config_file)
        manager.load_config()

        # Exact boundaries: micro [500, 2500), small [2500, 10000)
        assert manager.get_tier_for_capital(500) == "micro"
        assert manager.get_tier_for_capital(2500) == "small"  # Lower bound inclusive

    def test_default_config_path_resolution(self):
        """Should resolve default config path correctly."""
        manager = ConfigManager()  # No path provided

        # Should set path relative to project root
        assert manager.config_path is not None
        assert "adaptive_portfolio_config.json" in str(manager.config_path)
