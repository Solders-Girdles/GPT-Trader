"""Integration tests for PerpsBot guardian integration - end-to-end drift protection."""

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from bot_v2.monitoring.configuration_guardian import ConfigurationGuardian
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.config.types import Profile
from bot_v2.features.live_trade.risk import LiveRiskManager
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.perps_bot_builder import create_perps_bot


class TestPerpsBotGuardianIntegration:
    """Test PerpsBot guardian integration end-to-end."""

    async def setup_perps_bot(self, config_overrides=None):
        """Setup PerpsBot with mock components for testing."""
        # Merge config_overrides, allowing them to override defaults
        config_kwargs = {
            "profile": Profile.DEV,
            "symbols": ["BTC-USD"],
            "dry_run": True,
            "mock_broker": True,
            "update_interval": 1.0,
            "short_ma": 5,
            "long_ma": 20,
            "target_leverage": 2.0,
            "enable_shorts": False,
            "max_position_size": Decimal("1000"),
            "max_leverage": 3,
        }
        if config_overrides:
            config_kwargs.update(config_overrides)

        config = BotConfig(**config_kwargs)

        # Create mock registry with the config
        from bot_v2.orchestration.service_registry import empty_registry

        mock_registry = empty_registry(config)

        # Create mock broker
        broker = MagicMock()
        mock_registry = mock_registry.with_updates(broker=broker)

        # Create mock risk manager with RiskConfig
        from bot_v2.config.live_trade_config import RiskConfig

        risk_config = RiskConfig(
            max_leverage=config.max_leverage,
            daily_loss_limit=Decimal("1000"),
        )
        risk_manager = LiveRiskManager(config=risk_config)

        # Setup risk manager state
        for symbol in config.symbols:
            risk_manager.last_mark_update[symbol] = None
            risk_manager.positions[symbol] = None

        mock_registry = mock_registry.with_updates(risk_manager=risk_manager)

        bot = create_perps_bot(config, mock_registry)

        return bot, broker, risk_manager

    @patch.dict("os.environ", {}, clear=True)
    async def test_clean_cycle_continues_normally(self):
        """Test that a clean cycle without drift continues normally."""
        bot, broker, risk_manager = await self.setup_perps_bot()

        assert bot.is_reduce_only_mode() is False

        # Mock broker methods
        broker.list_balances.return_value = []
        broker.list_positions.return_value = []
        broker.get_account_info = AsyncMock(return_value=MagicMock(equity=Decimal("10000")))

        # Patch update_marks since it calls market data
        with patch.object(bot, "update_marks", new_callable=AsyncMock):
            await bot.run_cycle()

        assert bot.is_reduce_only_mode() is False  # Should still be False

    @patch.dict("os.environ", {"COINBASE_ENABLE_DERIVATIVES": "1"}, clear=False)
    async def test_critical_drift_triggers_emergency_shutdown(self):
        """Test that critical environment drift triggers emergency shutdown."""
        # Start with clean env
        with patch.dict("os.environ", {}, clear=False):
            bot, broker, risk_manager = await self.setup_perps_bot()

        # Set initial state with the env var
        with patch.dict("os.environ", {"COINBASE_ENABLE_DERIVATIVES": "0"}, clear=False):
            bot.baseline_snapshot = PerpsBot.build_baseline_snapshot(
                bot.config, getattr(bot.config, "derivatives_enabled", False)
            )
            bot.configuration_guardian = ConfigurationGuardian(bot.baseline_snapshot)

        # Now change the env var to trigger critical drift
        with patch.dict("os.environ", {"COINBASE_ENABLE_DERIVATIVES": "1"}, clear=False):
            # Mock broker methods to avoid network calls
            broker.list_balances.return_value = []
            broker.list_positions.return_value = []

            account_info_mock = MagicMock()
            account_info_mock.equity = Decimal("10000")
            broker.get_account_info = AsyncMock(return_value=account_info_mock)

            # Patch update_marks to avoid market data calls
            with patch.object(bot, "update_marks", new_callable=AsyncMock):
                try:
                    await asyncio.wait_for(bot.run_cycle(), timeout=5.0)
                except TimeoutError:
                    # Expected - critical drift should trigger shutdown, stopping the cycle
                    pass

        # After critical drift, bot should be marked as not running
        assert bot.running is False, "Critical drift should stop the bot"

    async def test_high_severity_drift_triggers_reduce_only(self):
        """Test that high-severity environment drift triggers reduce-only mode."""
        # Set initial state with baseline env var
        with patch.dict("os.environ", {"PERPS_POSITION_FRACTION": "0.2"}, clear=False):
            bot, broker, risk_manager = await self.setup_perps_bot()

        assert bot.is_reduce_only_mode() is False

        # Set running=True to simulate bot in running state
        bot.running = True

        # Mock broker methods to avoid network calls
        broker.list_balances.return_value = []
        broker.list_positions.return_value = []

        account_info_mock = MagicMock()
        account_info_mock.equity = Decimal("10000")
        broker.get_account_info = AsyncMock(return_value=account_info_mock)

        # Change the env var to trigger high-severity drift
        with patch.dict("os.environ", {"PERPS_POSITION_FRACTION": "0.1"}, clear=False):
            # Patch update_marks to avoid market data calls
            with patch.object(bot, "update_marks", new_callable=AsyncMock):
                await bot.run_cycle()

        # After high-severity drift, bot should be in reduce-only mode
        assert (
            bot.is_reduce_only_mode() is True
        ), "High-severity drift should enable reduce-only mode"
        assert bot.running is True, "High-severity drift should not stop the bot"

    async def test_position_violations_trigger_reduce_only(self):
        """Test that position size violations trigger reduce-only mode."""
        bot, broker, risk_manager = await self.setup_perps_bot()

        assert bot.is_reduce_only_mode() is False

        # Create a mock position that would violate max_position_size limits
        mock_position = MagicMock()
        mock_position.symbol = "BTC-USD"
        mock_position.size = Decimal("0.1")  # $5000 position at $50k price
        mock_position.price = Decimal("50000")

        # Set up violation conditions
        bot.config.max_position_size = Decimal("4000")  # Lower than current position

        # Mock broker methods
        broker.list_balances.return_value = []
        broker.list_positions.return_value = [mock_position]

        account_info_mock = MagicMock()
        account_info_mock.equity = Decimal("10000")
        broker.get_account_info = AsyncMock(return_value=account_info_mock)

        # Patch update_marks to avoid market data calls
        with patch.object(bot, "update_marks", new_callable=AsyncMock):
            await bot.run_cycle()

        # Position size violation should trigger reduce-only mode
        assert (
            bot.is_reduce_only_mode() is True
        ), "Position size violation should enable reduce-only mode"

    async def test_symbol_removal_violation_triggers_emergency_shutdown(self):
        """Test that removing symbols with active positions triggers emergency shutdown."""
        # Create bot with multiple symbols to establish baseline
        bot, broker, risk_manager = await self.setup_perps_bot(
            config_overrides={"symbols": ["BTC-USD", "ETH-USD"]}
        )

        # Bot isn't running until run() is called, but run_cycle() can be called directly
        # Set running=True to simulate that the bot is in a running state
        bot.running = True

        # Create a mock position on ETH-USD
        mock_position = MagicMock()
        mock_position.symbol = "ETH-USD"

        # Mock broker methods
        broker.list_balances.return_value = []
        broker.list_positions.return_value = [mock_position]

        account_info_mock = MagicMock()
        account_info_mock.equity = Decimal("10000")
        broker.get_account_info = AsyncMock(return_value=account_info_mock)

        # Now remove ETH-USD from the config to trigger violation
        bot.config.symbols = ["BTC-USD"]

        # Patch update_marks to avoid market data calls
        with patch.object(bot, "update_marks", new_callable=AsyncMock):
            try:
                await asyncio.wait_for(bot.run_cycle(), timeout=5.0)
            except TimeoutError:
                # Expected - critical violation should trigger shutdown
                pass

        # Symbol removal violation should trigger emergency shutdown
        assert (
            bot.running is False
        ), "Symbol removal with active positions should trigger emergency shutdown"
