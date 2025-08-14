"""Unit tests for the unified configuration system."""

import os
import json
import pytest
from pathlib import Path
from decimal import Decimal
from unittest.mock import patch, MagicMock
from datetime import datetime

from bot.config import (
    TradingConfig,
    FinancialConfig,
    Environment,
    DataConfig,
    LoggingConfig,
    AlpacaConfig,
    OptimizationConfig,
    get_config,
    set_config,
)
from bot.config.financial_config import (
    CapitalAllocation,
    TradingLimits,
    RiskParameters,
    TransactionCosts,
    OptimizationParameters,
)


class TestFinancialConfig:
    """Test suite for FinancialConfig."""

    def test_default_values(self):
        """Test default financial configuration values."""
        config = FinancialConfig()

        # Test capital defaults
        assert config.capital.initial_capital == Decimal("100000.0")
        assert config.capital.paper_trading_capital == Decimal("100000.0")
        assert config.capital.backtesting_capital == Decimal("100000.0")
        assert config.capital.deployment_budget == Decimal("10000.0")

        # Test limits defaults
        assert config.limits.max_position_size == Decimal("1000000.0")
        assert config.limits.max_order_value == Decimal("100000.0")
        assert config.limits.max_portfolio_positions == 20

        # Test risk defaults
        assert config.risk.max_portfolio_risk == 0.02
        assert config.risk.max_drawdown_percent == 0.20
        assert config.risk.stop_loss_percent == 0.02

        # Test costs defaults
        assert config.costs.commission_rate_bps == 10.0
        assert config.costs.commission_rate_decimal == 0.001
        assert config.costs.slippage_bps == 5.0

    def test_capital_validation(self):
        """Test capital amount validation."""
        with pytest.raises(ValueError):
            CapitalAllocation(initial_capital=Decimal("500"))  # Too low

        with pytest.raises(ValueError):
            CapitalAllocation(initial_capital=Decimal("200000000"))  # Too high

    def test_available_trading_capital(self):
        """Test available trading capital calculation."""
        config = FinancialConfig()
        config.capital.initial_capital = Decimal("100000")
        config.capital.reserve_capital_percent = 0.20

        available = config.capital.available_trading_capital
        assert available == Decimal("80000")

    def test_transaction_cost_calculations(self):
        """Test transaction cost property calculations."""
        config = FinancialConfig()
        config.costs.commission_rate_bps = 10.0
        config.costs.slippage_bps = 5.0
        config.costs.market_impact_bps = 2.0

        assert config.costs.commission_rate_decimal == 0.001
        assert config.costs.slippage_decimal == 0.0005
        assert config.costs.total_cost_bps == 17.0

    @patch.dict(
        os.environ,
        {
            "TRADING_INITIAL_CAPITAL": "250000",
            "TRADING_MAX_POSITION_SIZE": "50000",
            "TRADING_COMMISSION_BPS": "5",
        },
    )
    def test_from_env(self):
        """Test loading configuration from environment variables."""
        config = FinancialConfig.from_env()

        assert config.capital.initial_capital == Decimal("250000")
        assert config.limits.max_position_size == Decimal("50000")
        assert config.costs.commission_rate_bps == 5.0

    def test_save_and_load(self, tmp_path):
        """Test saving and loading configuration."""
        config1 = FinancialConfig()
        config1.capital.initial_capital = Decimal("150000")
        config1.costs.commission_rate_bps = 8.0

        # Save configuration
        config_path = tmp_path / "financial_config.json"
        config1.save_to_file(config_path)

        assert config_path.exists()

        # Load configuration
        config2 = FinancialConfig.load_from_file(config_path)

        assert config2.capital.initial_capital == Decimal("150000")
        assert config2.costs.commission_rate_bps == 8.0


class TestTradingConfig:
    """Test suite for unified TradingConfig."""

    def test_default_initialization(self):
        """Test default trading configuration."""
        config = TradingConfig()

        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is False
        assert config.enable_paper_trading is True
        assert config.enable_live_trading is False

        # Check sub-configurations
        assert isinstance(config.financial, FinancialConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.alpaca, AlpacaConfig)
        assert isinstance(config.optimization, OptimizationConfig)

    def test_environment_validation(self):
        """Test environment-specific validation."""
        # Cannot enable live trading in development
        with pytest.raises(ValueError):
            TradingConfig(environment=Environment.DEVELOPMENT, enable_live_trading=True)

        # Cannot enable both live and paper trading
        with pytest.raises(ValueError):
            TradingConfig(enable_live_trading=True, enable_paper_trading=True)

        # Cannot enable debug in production
        with pytest.raises(ValueError):
            TradingConfig(environment=Environment.PRODUCTION, debug=True)

    def test_profile_loading(self):
        """Test loading configuration profiles."""
        # Development profile
        config = TradingConfig.load(profile="dev")
        assert config.environment == Environment.DEVELOPMENT
        assert config.debug is True
        assert config.enable_paper_trading is True
        assert config.logging.level == "DEBUG"

        # Testing profile
        config = TradingConfig.load(profile="test")
        assert config.environment == Environment.TESTING
        assert config.debug is False
        assert config.dry_run is True
        assert config.logging.level == "INFO"

        # Production profile
        config = TradingConfig.load(profile="prod")
        assert config.environment == Environment.PRODUCTION
        assert config.debug is False
        assert config.enable_live_trading is True
        assert config.enable_notifications is True
        assert config.logging.structured_logging is True

    @patch("src.bot.config.unified_config.get_secret_manager")
    def test_env_override(self, mock_secret_manager):
        """Test environment variable overrides."""
        mock_manager = MagicMock()
        mock_manager.get_secret.side_effect = lambda key, default=None: {
            "ENVIRONMENT": "production",
            "DEBUG": "false",
            "LOG_LEVEL": "ERROR",
            "ENABLE_PAPER_TRADING": "false",
            "ENABLE_LIVE_TRADING": "true",
        }.get(key, default)
        mock_secret_manager.return_value = mock_manager

        config = TradingConfig.load()

        assert config.environment == Environment.PRODUCTION
        assert config.debug is False
        assert config.logging.level == "ERROR"
        assert config.enable_paper_trading is False
        assert config.enable_live_trading is True

    def test_get_initial_capital(self):
        """Test getting initial capital based on mode."""
        config = TradingConfig()

        # Paper trading mode
        config.enable_paper_trading = True
        config.enable_live_trading = False
        assert config.get_initial_capital() == float(config.financial.capital.paper_trading_capital)

        # Live trading mode
        config.enable_paper_trading = False
        config.enable_live_trading = True
        assert config.get_initial_capital() == float(config.financial.capital.initial_capital)

        # Backtesting mode (neither paper nor live)
        config.enable_paper_trading = False
        config.enable_live_trading = False
        assert config.get_initial_capital() == float(config.financial.capital.backtesting_capital)

    def test_save_and_load_config(self, tmp_path):
        """Test saving and loading full configuration."""
        config1 = TradingConfig()
        config1.environment = Environment.STAGING
        config1.financial.capital.initial_capital = Decimal("200000")
        config1.data.strict_validation = False
        config1.logging.level = "WARNING"

        # Save configuration
        config_path = tmp_path / "trading_config.json"
        config1.save(config_path)

        assert config_path.exists()

        # Load configuration
        config2 = TradingConfig.load(config_path)

        assert config2.environment == Environment.STAGING
        assert config2.financial.capital.initial_capital == Decimal("200000")
        assert config2.data.strict_validation is False
        assert config2.logging.level == "WARNING"

    def test_is_production_development(self):
        """Test environment helper properties."""
        config = TradingConfig()

        config.environment = Environment.PRODUCTION
        assert config.is_production is True
        assert config.is_development is False

        config.environment = Environment.DEVELOPMENT
        assert config.is_production is False
        assert config.is_development is True


class TestConfigSingleton:
    """Test configuration singleton pattern."""

    def test_get_config_singleton(self):
        """Test that get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_set_config(self):
        """Test setting global configuration."""
        custom_config = TradingConfig()
        custom_config.debug = True

        set_config(custom_config)

        retrieved_config = get_config()
        assert retrieved_config is custom_config
        assert retrieved_config.debug is True


class TestDataConfig:
    """Test data configuration."""

    def test_defaults(self):
        """Test default data configuration."""
        config = DataConfig()

        assert config.cache_dir == Path("data/cache")
        assert config.strict_validation is True
        assert config.default_source == "yfinance"
        assert config.max_cache_age_days == 7

    def test_data_source_validation(self):
        """Test data source validation."""
        with pytest.raises(ValueError):
            DataConfig(default_source="invalid_source")

        # Valid sources should work
        for source in ["yfinance", "alpaca", "polygon", "local"]:
            config = DataConfig(default_source=source)
            assert config.default_source == source


class TestLoggingConfig:
    """Test logging configuration."""

    def test_defaults(self):
        """Test default logging configuration."""
        config = LoggingConfig()

        assert config.level == "INFO"
        assert config.max_size_mb == 10
        assert config.backup_count == 5
        assert config.structured_logging is False

    def test_log_level_validation(self):
        """Test log level validation."""
        config = LoggingConfig(level="debug")
        assert config.level == "DEBUG"  # Should be uppercased

        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID")

    def test_max_size_bytes(self):
        """Test max size conversion to bytes."""
        config = LoggingConfig(max_size_mb=5)
        assert config.max_size_bytes == 5 * 1024 * 1024


class TestOptimizationParameters:
    """Test optimization parameters."""

    def test_defaults(self):
        """Test default optimization parameters."""
        params = OptimizationParameters()

        assert params.min_trades_for_validation == 30
        assert params.min_sharpe_ratio == 1.0
        assert params.target_sharpe_ratio == 2.0
        assert params.optimization_metric == "sharpe_ratio"

    def test_metric_validation(self):
        """Test optimization metric validation."""
        with pytest.raises(ValueError):
            OptimizationParameters(optimization_metric="invalid_metric")

        # Valid metrics should work
        for metric in ["sharpe_ratio", "total_return", "win_rate", "profit_factor"]:
            params = OptimizationParameters(optimization_metric=metric)
            assert params.optimization_metric == metric
