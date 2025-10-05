"""Tests for profile-based broker selection and credential validation."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.deterministic_broker import DeterministicBroker
from bot_v2.orchestration.runtime_coordinator import RuntimeCoordinator


@pytest.fixture
def mock_bot():
    """Create a mock bot with minimal required attributes."""
    bot = Mock()
    bot.config = BotConfig(profile=Profile.DEV, symbols=["BTC-USD"])
    bot.registry = Mock()
    bot.registry.broker = None
    bot.registry.risk_manager = None
    bot.registry.with_updates = Mock(return_value=bot.registry)
    bot.broker = None
    bot.risk_manager = None
    bot._product_map = {}
    bot.config_controller = Mock()
    bot.event_store = Mock()
    bot.strategy_orchestrator = Mock()
    bot.execution_coordinator = Mock()
    bot.metrics_server = None
    return bot


class TestProfileBasedBrokerSelection:
    """Test broker selection based on profile and environment variables."""

    def test_dev_profile_uses_deterministic_broker(self, mock_bot):
        """Dev profile should always use DeterministicBroker."""
        mock_bot.config.profile = Profile.DEV
        mock_bot.config.mock_broker = False

        coordinator = RuntimeCoordinator(mock_bot)
        coordinator._init_broker()

        assert isinstance(mock_bot.broker, DeterministicBroker)

    def test_demo_profile_uses_deterministic_broker(self, mock_bot):
        """Demo profile should always use DeterministicBroker."""
        mock_bot.config.profile = Profile.DEMO
        mock_bot.config.mock_broker = False

        coordinator = RuntimeCoordinator(mock_bot)
        coordinator._init_broker()

        assert isinstance(mock_bot.broker, DeterministicBroker)

    def test_mock_broker_flag_forces_deterministic(self, mock_bot):
        """mock_broker=True should force DeterministicBroker regardless of profile."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = True

        coordinator = RuntimeCoordinator(mock_bot)
        coordinator._init_broker()

        assert isinstance(mock_bot.broker, DeterministicBroker)

    @patch.dict(os.environ, {"BROKER": "coinbase"}, clear=True)
    def test_perps_force_mock_uses_deterministic(self, mock_bot):
        """perps_force_mock=True should use DeterministicBroker."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_force_mock = True

        coordinator = RuntimeCoordinator(mock_bot)
        coordinator._init_broker()

        assert isinstance(mock_bot.broker, DeterministicBroker)

    @patch.dict(os.environ, {"BROKER": "coinbase"}, clear=True)
    def test_perps_paper_trading_uses_deterministic(self, mock_bot):
        """perps_paper_trading=True should use DeterministicBroker."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.perps_paper_trading = True

        coordinator = RuntimeCoordinator(mock_bot)
        coordinator._init_broker()

        assert isinstance(mock_bot.broker, DeterministicBroker)

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "1",
            "COINBASE_SANDBOX_API_KEY": "test_key",
            "COINBASE_SANDBOX_API_SECRET": "test_secret",
            "COINBASE_SANDBOX_API_PASSPHRASE": "test_pass",
        },
        clear=True,
    )
    @patch("bot_v2.orchestration.runtime_coordinator.create_brokerage")
    def test_staging_with_sandbox_creates_coinbase(self, mock_create, mock_bot):
        """Staging profile with COINBASE_SANDBOX=1 should create CoinbaseBrokerage."""
        mock_bot.config.profile = Profile.STAGING
        mock_bot.config.mock_broker = False
        mock_bot.config.symbols = ["BTC-USD"]  # Spot symbol

        mock_broker = Mock()
        mock_broker.connect.return_value = True
        mock_broker.list_products.return_value = []
        mock_create.return_value = mock_broker

        coordinator = RuntimeCoordinator(mock_bot)
        coordinator._init_broker()

        mock_create.assert_called_once()
        assert mock_bot.broker is mock_broker

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_PROD_API_KEY": "prod_key",
            "COINBASE_PROD_API_SECRET": "prod_secret",
        },
        clear=True,
    )
    @patch("bot_v2.orchestration.runtime_coordinator.create_brokerage")
    def test_prod_profile_creates_coinbase(self, mock_create, mock_bot):
        """Production profile should create CoinbaseBrokerage with prod credentials."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.symbols = ["BTC-USD"]  # Spot symbol

        mock_broker = Mock()
        mock_broker.connect.return_value = True
        mock_broker.list_products.return_value = []
        mock_create.return_value = mock_broker

        coordinator = RuntimeCoordinator(mock_bot)
        coordinator._init_broker()

        mock_create.assert_called_once()
        assert mock_bot.broker is mock_broker

    def test_broker_from_registry_is_reused(self, mock_bot):
        """If broker exists in registry, it should be reused."""
        existing_broker = DeterministicBroker()
        mock_bot.registry.broker = existing_broker

        coordinator = RuntimeCoordinator(mock_bot)
        coordinator._init_broker()

        assert mock_bot.broker is existing_broker


class TestCredentialValidation:
    """Test credential validation for different profiles and environments."""

    @patch.dict(os.environ, {"BROKER": "coinbase"}, clear=True)
    def test_missing_broker_env_raises_error(self, mock_bot):
        """Missing BROKER env var should raise RuntimeError."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False

        with patch.dict(os.environ, {}, clear=True):
            coordinator = RuntimeCoordinator(mock_bot)
            with pytest.raises(RuntimeError, match="BROKER must be set"):
                coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "1",
        },
        clear=True,
    )
    def test_sandbox_on_prod_profile_raises_error(self, mock_bot):
        """COINBASE_SANDBOX=1 on production profile should raise RuntimeError."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False

        coordinator = RuntimeCoordinator(mock_bot)
        with pytest.raises(RuntimeError, match="COINBASE_SANDBOX=1 is only allowed"):
            coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "1",
        },
        clear=True,
    )
    def test_sandbox_on_canary_profile_raises_error(self, mock_bot):
        """COINBASE_SANDBOX=1 on canary profile should raise RuntimeError."""
        mock_bot.config.profile = Profile.CANARY
        mock_bot.config.mock_broker = False

        coordinator = RuntimeCoordinator(mock_bot)
        with pytest.raises(RuntimeError, match="COINBASE_SANDBOX=1 is only allowed"):
            coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "1",
        },
        clear=True,
    )
    def test_sandbox_on_staging_allowed(self, mock_bot):
        """COINBASE_SANDBOX=1 on staging profile should be allowed."""
        mock_bot.config.profile = Profile.STAGING
        mock_bot.config.mock_broker = False
        mock_bot.config.symbols = ["BTC-USD"]

        # Missing credentials should raise, but not sandbox validation
        coordinator = RuntimeCoordinator(mock_bot)
        with pytest.raises(RuntimeError, match="Sandbox spot trading"):
            coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "1",
        },
        clear=True,
    )
    def test_sandbox_spot_missing_credentials_raises(self, mock_bot):
        """Sandbox spot trading without credentials should raise clear error."""
        mock_bot.config.profile = Profile.STAGING
        mock_bot.config.mock_broker = False
        mock_bot.config.symbols = ["BTC-USD"]

        coordinator = RuntimeCoordinator(mock_bot)
        with pytest.raises(RuntimeError, match="Sandbox spot trading.*requires credentials"):
            coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "1",
            "COINBASE_SANDBOX_API_KEY": "key",
            "COINBASE_SANDBOX_API_SECRET": "secret",
            "COINBASE_SANDBOX_API_PASSPHRASE": "pass",
        },
        clear=True,
    )
    def test_sandbox_spot_with_credentials_passes(self, mock_bot):
        """Sandbox spot trading with all credentials should pass validation."""
        mock_bot.config.profile = Profile.STAGING
        mock_bot.config.mock_broker = False
        mock_bot.config.symbols = ["BTC-USD"]

        coordinator = RuntimeCoordinator(mock_bot)
        # Should not raise
        coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
        },
        clear=True,
    )
    def test_prod_spot_missing_credentials_raises(self, mock_bot):
        """Production spot trading without credentials should raise clear error."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.symbols = ["BTC-USD"]

        coordinator = RuntimeCoordinator(mock_bot)
        with pytest.raises(RuntimeError, match="Production spot trading.*requires credentials"):
            coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_PROD_API_KEY": "key",
            "COINBASE_PROD_API_SECRET": "secret",
        },
        clear=True,
    )
    def test_prod_spot_with_credentials_passes(self, mock_bot):
        """Production spot trading with credentials should pass validation."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.symbols = ["BTC-USD"]

        coordinator = RuntimeCoordinator(mock_bot)
        # Should not raise
        coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "1",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        },
        clear=True,
    )
    def test_sandbox_perps_missing_credentials_raises(self, mock_bot):
        """Sandbox perpetuals without credentials should raise clear error."""
        mock_bot.config.profile = Profile.STAGING
        mock_bot.config.mock_broker = False
        mock_bot.config.derivatives_enabled = True
        mock_bot.config.symbols = ["BTC-PERP"]

        coordinator = RuntimeCoordinator(mock_bot)
        with pytest.raises(
            RuntimeError, match="Sandbox perpetuals.*require Exchange API credentials"
        ):
            coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_SANDBOX": "1",
            "COINBASE_ENABLE_DERIVATIVES": "1",
            "COINBASE_SANDBOX_API_KEY": "key",
            "COINBASE_SANDBOX_API_SECRET": "secret",
            "COINBASE_SANDBOX_API_PASSPHRASE": "pass",
        },
        clear=True,
    )
    def test_sandbox_perps_with_credentials_passes(self, mock_bot):
        """Sandbox perpetuals with Exchange API credentials should pass."""
        mock_bot.config.profile = Profile.STAGING
        mock_bot.config.mock_broker = False
        mock_bot.config.derivatives_enabled = True
        mock_bot.config.symbols = ["BTC-PERP"]

        coordinator = RuntimeCoordinator(mock_bot)
        # Should not raise
        coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_ENABLE_DERIVATIVES": "1",
        },
        clear=True,
    )
    def test_prod_perps_missing_cdp_credentials_raises(self, mock_bot):
        """Production perpetuals without CDP credentials should raise clear error."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.derivatives_enabled = True
        mock_bot.config.symbols = ["BTC-PERP"]

        coordinator = RuntimeCoordinator(mock_bot)
        with pytest.raises(
            RuntimeError, match="Production perpetuals.*require CDP JWT credentials"
        ):
            coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_ENABLE_DERIVATIVES": "1",
            "COINBASE_PROD_CDP_API_KEY": "cdp_key",
            "COINBASE_PROD_CDP_PRIVATE_KEY": "cdp_private",
        },
        clear=True,
    )
    def test_prod_perps_with_cdp_credentials_passes(self, mock_bot):
        """Production perpetuals with CDP credentials should pass validation."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.derivatives_enabled = True
        mock_bot.config.symbols = ["BTC-PERP"]

        coordinator = RuntimeCoordinator(mock_bot)
        # Should not raise
        coordinator._validate_broker_environment()

    @patch.dict(
        os.environ,
        {
            "BROKER": "coinbase",
            "COINBASE_ENABLE_DERIVATIVES": "1",
            "COINBASE_API_MODE": "exchange",
        },
        clear=True,
    )
    def test_prod_perps_with_exchange_mode_raises(self, mock_bot):
        """Production perpetuals with exchange mode should raise error."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.mock_broker = False
        mock_bot.config.derivatives_enabled = True
        mock_bot.config.symbols = ["BTC-PERP"]

        coordinator = RuntimeCoordinator(mock_bot)
        with pytest.raises(RuntimeError, match="Production perpetuals.*require Advanced Trade API"):
            coordinator._validate_broker_environment()

    def test_paper_mode_skips_validation(self, mock_bot):
        """Paper trading mode should skip production env checks."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.perps_paper_trading = True

        coordinator = RuntimeCoordinator(mock_bot)
        # Should not raise even with missing credentials
        coordinator._validate_broker_environment()

    def test_force_mock_skips_validation(self, mock_bot):
        """Force mock mode should skip production env checks."""
        mock_bot.config.profile = Profile.PROD
        mock_bot.config.perps_force_mock = True

        coordinator = RuntimeCoordinator(mock_bot)
        # Should not raise even with missing credentials
        coordinator._validate_broker_environment()
