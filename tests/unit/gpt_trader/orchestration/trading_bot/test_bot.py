"""Comprehensive tests for TradingBot."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gpt_trader.orchestration.trading_bot.bot import TradingBot


class TestTradingBotInitialization:
    """Test TradingBot initialization."""

    @pytest.fixture
    def mock_config(self) -> Mock:
        """Create a mock BotConfig."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC", "ETH-PERP-USDC"]
        config.interval = 60
        return config

    @pytest.fixture
    def mock_registry(self) -> Mock:
        """Create a mock service registry."""
        registry = Mock()
        registry.broker = Mock()
        registry.account_manager = Mock()
        registry.account_telemetry = Mock()
        registry.risk_manager = Mock()
        registry.runtime_state = Mock()
        registry.event_store = Mock()
        registry.notification_service = Mock()
        return registry

    def test_init_with_config_only(self, mock_config: Mock) -> None:
        """Test initialization with only config."""
        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config)

        assert bot.config is mock_config
        assert bot.container is None
        assert bot.running is False
        assert bot.broker is None
        assert bot.account_manager is None
        assert bot.risk_manager is None
        assert bot.runtime_state is None

    def test_init_with_registry(self, mock_config: Mock, mock_registry: Mock) -> None:
        """Test initialization with registry."""
        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, registry=mock_registry)

        assert bot.broker is mock_registry.broker
        assert bot.account_manager is mock_registry.account_manager
        assert bot.account_telemetry is mock_registry.account_telemetry
        assert bot.risk_manager is mock_registry.risk_manager
        assert bot.runtime_state is mock_registry.runtime_state

    def test_init_with_container(self, mock_config: Mock) -> None:
        """Test initialization with container."""
        mock_container = Mock()

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, container=mock_container)

        assert bot.container is mock_container

    def test_init_creates_context(self, mock_config: Mock, mock_registry: Mock) -> None:
        """Test that initialization creates CoordinatorContext."""
        with (
            patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine,
            patch("gpt_trader.orchestration.trading_bot.bot.CoordinatorContext") as mock_context,
        ):
            mock_engine.return_value = Mock()
            mock_context.return_value = Mock()

            bot = TradingBot(config=mock_config, registry=mock_registry)

            mock_context.assert_called_once_with(
                config=mock_config,
                registry=mock_registry,
                broker=mock_registry.broker,
                symbols=tuple(mock_config.symbols),
                risk_manager=mock_registry.risk_manager,
                event_store=mock_registry.event_store,
                notification_service=mock_registry.notification_service,
            )
            assert bot.context is not None

    def test_init_creates_engine(self, mock_config: Mock) -> None:
        """Test that initialization creates TradingEngine."""
        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config)

            assert bot.engine is not None
            mock_engine.assert_called_once()

    def test_init_registry_missing_attributes(self, mock_config: Mock) -> None:
        """Test initialization handles registry with missing attributes."""
        minimal_registry = Mock(spec=["broker"])
        minimal_registry.broker = Mock()

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, registry=minimal_registry)

        assert bot.broker is minimal_registry.broker
        assert bot.account_manager is None
        assert bot.risk_manager is None


class TestTradingBotRun:
    """Test TradingBot run method."""

    @pytest.fixture
    def bot(self) -> TradingBot:
        """Create a TradingBot with mocked engine."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 1  # Short interval for testing

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.start_background_tasks = AsyncMock(return_value=[])
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config)

        return bot

    @pytest.mark.asyncio
    async def test_run_sets_running_flag(self, bot: TradingBot) -> None:
        """Test that run sets running flag."""
        await bot.run(single_cycle=True)

        # After completion, running should be False
        assert bot.running is False

    @pytest.mark.asyncio
    async def test_run_single_cycle_calls_shutdown(self, bot: TradingBot) -> None:
        """Test that single_cycle mode calls shutdown."""
        await bot.run(single_cycle=True)

        bot.engine.shutdown.assert_called()

    @pytest.mark.asyncio
    async def test_run_starts_background_tasks(self, bot: TradingBot) -> None:
        """Test that run starts background tasks."""
        await bot.run(single_cycle=True)

        bot.engine.start_background_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_handles_cancellation(self, bot: TradingBot) -> None:
        """Test that run handles CancelledError gracefully."""

        # Make start_background_tasks return a task that gets cancelled
        async def cancelled_task() -> None:
            raise asyncio.CancelledError()

        bot.engine.start_background_tasks = AsyncMock(
            return_value=[asyncio.create_task(cancelled_task())]
        )

        # Should not raise
        await bot.run(single_cycle=False)

        # Shutdown should still be called in finally
        bot.engine.shutdown.assert_called()
        assert bot.running is False


class TestTradingBotStop:
    """Test TradingBot stop/shutdown methods."""

    @pytest.fixture
    def bot(self) -> TradingBot:
        """Create a TradingBot with mocked engine."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config)
            bot.running = True

        return bot

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, bot: TradingBot) -> None:
        """Test that stop sets running to False."""
        await bot.stop()

        assert bot.running is False

    @pytest.mark.asyncio
    async def test_stop_calls_engine_shutdown(self, bot: TradingBot) -> None:
        """Test that stop calls engine shutdown."""
        await bot.stop()

        bot.engine.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_is_alias_for_stop(self, bot: TradingBot) -> None:
        """Test that shutdown is an alias for stop."""
        await bot.shutdown()

        assert bot.running is False
        bot.engine.shutdown.assert_called_once()


class TestTradingBotExecuteDecision:
    """Test TradingBot execute_decision method."""

    @pytest.fixture
    def bot(self) -> TradingBot:
        """Create a TradingBot with mocked engine."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            engine = Mock()
            engine.execute_decision = Mock(return_value="order-123")
            mock_engine.return_value = engine
            bot = TradingBot(config=config)

        return bot

    def test_execute_decision_delegates_to_engine(self, bot: TradingBot) -> None:
        """Test that execute_decision delegates to engine."""
        decision = Mock()
        mark = Mock()
        product = Mock()
        position_state = Mock()

        result = bot.execute_decision(
            symbol="BTC-PERP-USDC",
            decision=decision,
            mark=mark,
            product=product,
            position_state=position_state,
        )

        assert result == "order-123"
        bot.engine.execute_decision.assert_called_once_with(
            "BTC-PERP-USDC", decision, mark, product, position_state
        )

    def test_execute_decision_returns_none_when_engine_lacks_method(self) -> None:
        """Test execute_decision returns None if engine lacks the method."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            engine = Mock(spec=[])  # No execute_decision
            mock_engine.return_value = engine
            bot = TradingBot(config=config)

        result = bot.execute_decision(
            symbol="BTC-PERP-USDC",
            decision=Mock(),
        )

        assert result is None

    def test_execute_decision_with_minimal_args(self, bot: TradingBot) -> None:
        """Test execute_decision with minimal arguments."""
        decision = Mock()

        result = bot.execute_decision(
            symbol="BTC-PERP-USDC",
            decision=decision,
        )

        assert result == "order-123"
        bot.engine.execute_decision.assert_called_once_with(
            "BTC-PERP-USDC", decision, None, None, None
        )


class TestTradingBotGetProduct:
    """Test TradingBot get_product method."""

    def test_get_product_with_broker(self) -> None:
        """Test get_product when broker is available."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60

        registry = Mock()
        mock_product = Mock()
        registry.broker = Mock()
        registry.broker.get_product = Mock(return_value=mock_product)

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=config, registry=registry)

        result = bot.get_product("BTC-PERP-USDC")

        assert result is mock_product
        registry.broker.get_product.assert_called_once_with("BTC-PERP-USDC")

    def test_get_product_without_broker(self) -> None:
        """Test get_product when broker is None."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=config)

        result = bot.get_product("BTC-PERP-USDC")

        assert result is None

    def test_get_product_broker_without_method(self) -> None:
        """Test get_product when broker lacks get_product method."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60

        registry = Mock()
        registry.broker = Mock(spec=[])  # No get_product

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=config, registry=registry)

        result = bot.get_product("BTC-PERP-USDC")

        assert result is None


class TestTradingBotStateManagement:
    """Test TradingBot state management."""

    def test_initial_state(self) -> None:
        """Test initial state of TradingBot."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=config)

        assert bot.running is False

    @pytest.mark.asyncio
    async def test_running_state_during_execution(self) -> None:
        """Test running state changes during execution."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 0.01  # Very short for testing

        running_states: list[bool] = []

        async def capture_state() -> list:
            # Capture state immediately after run starts
            await asyncio.sleep(0.001)
            running_states.append(bot.running)
            return []

        with patch("gpt_trader.orchestration.trading_bot.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.start_background_tasks = capture_state
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config)

        # Before run
        assert bot.running is False

        await bot.run(single_cycle=True)

        # During run (captured by capture_state)
        assert running_states == [True]

        # After run
        assert bot.running is False
