"""Comprehensive tests for TradingBot."""

from __future__ import annotations

import asyncio
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest

from gpt_trader.features.live_trade.bot import TradingBot


class TestTradingBotInitialization:
    """Test TradingBot initialization."""

    @pytest.fixture
    def mock_config(self) -> Mock:
        """Create a mock BotConfig."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC", "ETH-PERP-USDC"]
        config.interval = 60
        return config

    def test_init_with_config_only(self, mock_config: Mock) -> None:
        """Test initialization with only config."""
        mock_container = Mock()
        # Set container attributes to None for this test
        mock_container.broker = None
        mock_container.risk_manager = None
        mock_container.event_store = Mock()
        mock_container.orders_store = Mock()
        mock_container.notification_service = Mock()

        mock_container.account_manager = Mock()
        mock_container.account_telemetry = Mock()
        mock_container.runtime_state = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, container=mock_container)

        assert bot.broker is mock_container.broker
        assert bot.account_manager is mock_container.account_manager
        assert bot.account_telemetry is mock_container.account_telemetry
        assert bot.risk_manager is mock_container.risk_manager
        assert bot.runtime_state is mock_container.runtime_state

    def test_init_with_container(self, mock_config: Mock) -> None:
        """Test initialization with container."""
        mock_container = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, container=mock_container)

        assert bot.container is mock_container

    def test_init_creates_context(self, mock_config: Mock) -> None:
        """Test that initialization creates CoordinatorContext."""
        mock_container = Mock()
        mock_container.broker = Mock()
        mock_container.risk_manager = Mock()
        mock_container.event_store = Mock()
        mock_container.orders_store = Mock()
        mock_container.notification_service = Mock()

        with (
            patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine,
            patch("gpt_trader.features.live_trade.bot.CoordinatorContext") as mock_context,
        ):
            mock_engine.return_value = Mock()
            mock_context.return_value = Mock()

            bot = TradingBot(config=mock_config, container=mock_container)

            mock_context.assert_called_once_with(
                config=mock_config,
                container=mock_container,
                broker=mock_container.broker,
                broker_calls=ANY,
                symbols=tuple(mock_config.symbols),
                risk_manager=mock_container.risk_manager,
                event_store=mock_container.event_store,
                orders_store=mock_container.orders_store,
                notification_service=mock_container.notification_service,
            )
            assert bot.context is not None

    def test_init_creates_engine(self, mock_config: Mock) -> None:
        """Test that initialization creates TradingEngine."""
        mock_container = Mock()
        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, container=mock_container)

            assert bot.engine is not None
            mock_engine.assert_called_once()

    def test_init_container_missing_optional_attributes(self, mock_config: Mock) -> None:
        """Test initialization handles container with missing optional attributes."""
        from types import SimpleNamespace

        # Use SimpleNamespace with only required attributes
        mock_container = SimpleNamespace(
            broker=Mock(),
            risk_manager=Mock(),
            event_store=Mock(),
            orders_store=None,
            notification_service=Mock(),
            # Optional attributes not set - getattr will return None
        )

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=mock_config, container=mock_container)

        assert bot.broker is mock_container.broker
        assert bot.account_manager is None  # Not set in SimpleNamespace
        assert bot.account_telemetry is None  # Not set in SimpleNamespace
        assert bot.runtime_state is None  # Not set in SimpleNamespace


class TestTradingBotRun:
    """Test TradingBot run method."""

    @pytest.fixture
    def bot(self) -> TradingBot:
        """Create a TradingBot with mocked engine."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 1  # Short interval for testing
        mock_container = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.start_background_tasks = AsyncMock(return_value=[])
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config, container=mock_container)

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
        mock_container = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config, container=mock_container)
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
        mock_container = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            engine = Mock()
            engine.execute_decision = Mock(return_value="order-123")
            mock_engine.return_value = engine
            bot = TradingBot(config=config, container=mock_container)

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
        mock_container = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            engine = Mock(spec=[])  # No execute_decision
            mock_engine.return_value = engine
            bot = TradingBot(config=config, container=mock_container)

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
        mock_container = Mock()

        mock_product = Mock()
        mock_container.broker = Mock()
        mock_container.broker.get_product = Mock(return_value=mock_product)
        mock_container.risk_manager = Mock()
        mock_container.event_store = Mock()
        mock_container.notification_service = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=config, container=mock_container)

        result = bot.get_product("BTC-PERP-USDC")

        assert result is mock_product
        mock_container.broker.get_product.assert_called_once_with("BTC-PERP-USDC")

    def test_get_product_without_broker(self) -> None:
        """Test get_product when broker is None."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60
        mock_container = Mock()

        # Set container.broker to None
        mock_container.broker = None
        mock_container.risk_manager = Mock()
        mock_container.event_store = Mock()
        mock_container.notification_service = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=config, container=mock_container)

        result = bot.get_product("BTC-PERP-USDC")

        assert result is None

    def test_get_product_broker_without_method(self) -> None:
        """Test get_product when broker lacks get_product method."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60
        mock_container = Mock()

        mock_container.broker = Mock(spec=[])  # No get_product
        mock_container.risk_manager = Mock()
        mock_container.event_store = Mock()
        mock_container.notification_service = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=config, container=mock_container)

        result = bot.get_product("BTC-PERP-USDC")

        assert result is None


class TestTradingBotStateManagement:
    """Test TradingBot state management."""

    def test_initial_state(self) -> None:
        """Test initial state of TradingBot."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60
        mock_container = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=config, container=mock_container)

        assert bot.running is False

    @pytest.mark.asyncio
    async def test_running_state_during_execution(self) -> None:
        """Test running state changes during execution."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 0.01  # Very short for testing
        mock_container = Mock()

        running_states: list[bool] = []

        async def capture_state() -> list:
            # Capture state immediately after run starts
            await asyncio.sleep(0.001)
            running_states.append(bot.running)
            return []

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.start_background_tasks = capture_state
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config, container=mock_container)

        # Before run
        assert bot.running is False

        await bot.run(single_cycle=True)

        # During run (captured by capture_state)
        assert running_states == [True]

        # After run
        assert bot.running is False


class TestTradingBotFlattenAndStop:
    """Tests for the emergency flatten-and-stop flow."""

    @pytest.mark.asyncio
    async def test_flatten_and_stop_closes_positions_and_shuts_down(self) -> None:
        from decimal import Decimal
        from types import SimpleNamespace
        from unittest.mock import Mock

        from gpt_trader.app.config import BotConfig
        from gpt_trader.core import OrderSide, OrderType

        class _DirectBrokerCalls:
            async def __call__(self, fn, *args, **kwargs):
                return fn(*args, **kwargs)

        config = BotConfig(symbols=["BTC-USD"], interval=1)
        broker = Mock()
        broker.list_positions.return_value = [
            SimpleNamespace(symbol="BTC-USD", quantity=Decimal("1"))
        ]
        broker.place_order = Mock()

        container = SimpleNamespace(
            broker=broker,
            risk_manager=Mock(),
            event_store=Mock(),
            orders_store=Mock(),
            notification_service=Mock(),
        )

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config, container=container)

        bot._broker_calls = _DirectBrokerCalls()

        messages = await bot.flatten_and_stop()

        assert any("Submitted CLOSE for BTC-USD" in msg for msg in messages)
        broker.list_positions.assert_called_once()
        broker.place_order.assert_called_once_with(
            "BTC-USD",
            OrderSide.SELL,
            OrderType.MARKET,
            Decimal("1"),
        )
        engine.shutdown.assert_called_once()
        assert bot.running is False
