"""Tests for execution coordinator"""

import asyncio
import pytest
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.core.interfaces import Order, OrderSide, OrderType, Product
from bot_v2.orchestration.execution_coordinator import ExecutionCoordinator


@pytest.fixture
def mock_bot():
    """Mock PerpsBot instance"""
    bot = Mock()
    bot.config = Mock()
    bot.config.dry_run = False
    bot.config.enable_order_preview = False
    bot.broker = Mock()
    bot.risk_manager = Mock()
    bot.risk_manager.config = Mock()
    bot.risk_manager.config.kill_switch_enabled = False
    bot.event_store = Mock()
    bot.orders_store = Mock()
    bot.registry = Mock()
    bot.registry.extras = {}
    bot.registry.with_updates = Mock(return_value=bot.registry)
    bot.exec_engine = Mock()
    bot._order_lock = None
    bot.order_stats = {"attempted": 0, "successful": 0, "failed": 0}
    bot._product_map = {}
    bot.running = True
    bot.bot_id = "test_bot"
    return bot


@pytest.fixture
def coordinator(mock_bot):
    """Create ExecutionCoordinator instance"""
    return ExecutionCoordinator(mock_bot)


@pytest.fixture
def sample_product():
    """Create sample product"""
    product = Mock(spec=Product)
    product.symbol = "BTC-USD"
    product.base_increment = Decimal("0.00000001")
    product.quote_increment = Decimal("0.01")
    return product


@pytest.fixture
def sample_decision():
    """Create sample trading decision"""
    # Import locally to avoid module-level dependency on live_trade
    from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

    return Decision(
        action=Action.BUY,
        reason="test_signal",
        target_notional=Decimal("1000"),
        leverage=1,
        reduce_only=False,
    )


class TestExecutionCoordinator:
    """Test suite for ExecutionCoordinator"""

    def test_initialization(self, coordinator, mock_bot):
        """Test coordinator initialization"""
        assert coordinator._bot == mock_bot
        assert coordinator._order_placement_service is None
        assert coordinator._runtime_supervisor is None

    def test_init_execution_basic(self, coordinator, mock_bot):
        """Test basic execution initialization"""
        coordinator.init_execution()

        assert mock_bot.exec_engine is not None
        mock_bot.registry.with_updates.assert_called()

    @patch.dict("os.environ", {"SLIPPAGE_MULTIPLIERS": "BTC-USD:1.5,ETH-USD:2.0"})
    def test_init_execution_with_slippage(self, coordinator, mock_bot):
        """Test execution initialization with slippage multipliers"""
        coordinator.init_execution()

        # Should parse slippage configuration
        assert mock_bot.exec_engine is not None

    def test_init_execution_with_advanced_engine(self, coordinator, mock_bot):
        """Test initialization with advanced execution engine"""
        mock_bot.risk_manager.config.enable_dynamic_position_sizing = True

        coordinator.init_execution()

        # Should create AdvancedExecutionEngine
        assert mock_bot.exec_engine is not None

    @pytest.mark.asyncio
    async def test_execute_decision_dry_run(
        self, coordinator, mock_bot, sample_decision, sample_product
    ):
        """Test decision execution in dry run mode"""
        mock_bot.config.dry_run = True

        await coordinator.execute_decision(
            symbol="BTC-USD",
            decision=sample_decision,
            mark=Decimal("50000"),
            product=sample_product,
            position_state=None,
        )

        # Should not place actual order
        assert mock_bot.order_stats["attempted"] == 0

    @pytest.mark.asyncio
    async def test_execute_decision_buy(
        self, coordinator, mock_bot, sample_decision, sample_product
    ):
        """Test BUY decision execution"""
        mock_bot.is_reduce_only_mode = Mock(return_value=False)
        mock_bot.config.time_in_force = "GTC"

        # Mock the service's execute_decision method
        with patch(
            "bot_v2.orchestration.execution_coordinator.OrderPlacementService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            # Force re-creation of service
            coordinator._order_placement_service = None

            await coordinator.execute_decision(
                symbol="BTC-USD",
                decision=sample_decision,
                mark=Decimal("50000"),
                product=sample_product,
                position_state=None,
            )

            # Verify service was called
            mock_service.execute_decision.assert_called_once()
            call_kwargs = mock_service.execute_decision.call_args[1]
            assert call_kwargs["symbol"] == "BTC-USD"

    @pytest.mark.asyncio
    async def test_execute_decision_close_position(self, coordinator, mock_bot, sample_product):
        """Test CLOSE action execution"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        close_decision = Decision(
            action=Action.CLOSE,
            reason="take_profit",
            leverage=1,
            reduce_only=True,
        )

        position_state = {
            "quantity": Decimal("0.5"),
            "side": "long",
            "entry_price": Decimal("48000"),
        }

        mock_bot.is_reduce_only_mode = Mock(return_value=False)
        mock_bot.config.time_in_force = "GTC"

        # Mock the service's execute_decision method
        with patch(
            "bot_v2.orchestration.execution_coordinator.OrderPlacementService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            coordinator._order_placement_service = None

            await coordinator.execute_decision(
                symbol="BTC-USD",
                decision=close_decision,
                mark=Decimal("52000"),
                product=sample_product,
                position_state=position_state,
            )

            # Verify service was called
            mock_service.execute_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_decision_close_no_position(self, coordinator, mock_bot, sample_product):
        """Test CLOSE action with no position"""
        from bot_v2.features.live_trade.strategies.perps_baseline import Action, Decision

        close_decision = Decision(
            action=Action.CLOSE,
            reason="stop_loss",
            leverage=1,
            reduce_only=True,
        )

        coordinator._place_order = AsyncMock()

        mock_bot.is_reduce_only_mode = Mock(return_value=False)
        mock_bot.config.time_in_force = "GTC"

        await coordinator.execute_decision(
            symbol="BTC-USD",
            decision=close_decision,
            mark=Decimal("50000"),
            product=sample_product,
            position_state=None,
        )

        # Should not crash (warning logged instead)

    @pytest.mark.asyncio
    async def test_execute_decision_missing_product(self, coordinator, mock_bot, sample_decision):
        """Test decision execution with missing product"""
        mock_bot.is_reduce_only_mode = Mock(return_value=False)
        mock_bot.config.time_in_force = "GTC"

        with pytest.raises(AssertionError, match="Missing product metadata"):
            await coordinator.execute_decision(
                symbol="BTC-USD",
                decision=sample_decision,
                mark=Decimal("50000"),
                product=None,
                position_state=None,
            )

    @pytest.mark.asyncio
    async def test_execute_decision_invalid_mark(
        self, coordinator, mock_bot, sample_decision, sample_product
    ):
        """Test decision execution with invalid mark price"""
        mock_bot.is_reduce_only_mode = Mock(return_value=False)
        mock_bot.config.time_in_force = "GTC"

        with pytest.raises(AssertionError, match="Invalid mark"):
            await coordinator.execute_decision(
                symbol="BTC-USD",
                decision=sample_decision,
                mark=Decimal("0"),
                product=sample_product,
                position_state=None,
            )

    @pytest.mark.asyncio
    async def test_place_order_success(self, coordinator, mock_bot):
        """Test successful order placement delegation"""
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.symbol = "BTC-USD"
        mock_order.side = OrderSide.BUY
        mock_order.quantity = Decimal("0.02")

        # Mock the service
        with patch(
            "bot_v2.orchestration.execution_coordinator.OrderPlacementService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service._place_order_inner = AsyncMock(return_value=mock_order)
            mock_service_class.return_value = mock_service

            coordinator._order_placement_service = None

            result = await coordinator._place_order(
                mock_bot.exec_engine,
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("0.02"),
                order_type=OrderType.MARKET,
            )

            assert result == mock_order
            mock_service._place_order_inner.assert_called_once()
        assert mock_bot.order_stats["failed"] == 0

    @pytest.mark.asyncio
    async def test_place_order_validation_error(self, coordinator, mock_bot):
        """Test order placement with validation error"""
        coordinator._place_order_inner = AsyncMock(side_effect=ValidationError("Invalid quantity"))

        with pytest.raises(ValidationError):
            await coordinator._place_order(
                mock_bot.exec_engine,
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("0.02"),
            )

        assert mock_bot.order_stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_place_order_execution_error(self, coordinator, mock_bot):
        """Test order placement with execution error"""
        coordinator._place_order_inner = AsyncMock(
            side_effect=ExecutionError("Exchange rejected order")
        )

        with pytest.raises(ExecutionError):
            await coordinator._place_order(
                mock_bot.exec_engine,
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("0.02"),
            )

        assert mock_bot.order_stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_place_order_generic_exception(self, coordinator, mock_bot):
        """Test order placement with generic exception"""
        coordinator._place_order_inner = AsyncMock(side_effect=Exception("Unexpected error"))

        result = await coordinator._place_order(
            mock_bot.exec_engine,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.02"),
        )

        assert result is None
        assert mock_bot.order_stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_place_order_inner_increments_stats(self, coordinator, mock_bot):
        """Test that order placement increments stats"""
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.symbol = "BTC-USD"
        mock_order.quantity = Decimal("0.02")
        mock_order.side = OrderSide.BUY

        mock_bot.exec_engine.place_order = Mock(return_value="order_123")
        mock_bot.broker.get_order = Mock(return_value=mock_order)

        result = await coordinator._place_order_inner(
            mock_bot.exec_engine,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.02"),
        )

        assert mock_bot.order_stats["attempted"] == 1
        assert mock_bot.order_stats["successful"] == 1
        assert result == mock_order

    def test_ensure_order_lock_creates_lock(self, coordinator, mock_bot):
        """Test order lock creation"""
        lock = coordinator._ensure_order_lock()

        assert lock is not None
        assert isinstance(lock, asyncio.Lock)
        assert mock_bot._order_lock is not None

    def test_ensure_order_lock_reuses_existing(self, coordinator, mock_bot):
        """Test order lock reuse"""
        lock1 = coordinator._ensure_order_lock()
        lock2 = coordinator._ensure_order_lock()

        assert lock1 is lock2

    def test_get_order_reconciler_creates_instance(self, coordinator, mock_bot):
        """Test order reconciler creation"""
        reconciler = coordinator._get_order_reconciler()

        assert reconciler is not None
        assert coordinator._order_reconciler is not None

    def test_get_order_reconciler_reuses_instance(self, coordinator):
        """Test order reconciler reuse"""
        reconciler1 = coordinator._get_order_reconciler()
        reconciler2 = coordinator._get_order_reconciler()

        assert reconciler1 is reconciler2

    def test_reset_order_reconciler(self, coordinator):
        """Test order reconciler reset"""
        coordinator._get_order_reconciler()
        assert coordinator._order_reconciler is not None

        coordinator.reset_order_reconciler()
        assert coordinator._order_reconciler is None

    @pytest.mark.asyncio
    async def test_run_runtime_guards(self, coordinator, mock_bot):
        """Test runtime guards execution"""
        mock_bot.exec_engine.run_runtime_guards = Mock()

        # Run for short time then stop
        task = asyncio.create_task(coordinator.run_runtime_guards())
        await asyncio.sleep(0.1)
        mock_bot.running = False
        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_run_order_reconciliation(self, coordinator, mock_bot):
        """Test order reconciliation loop"""
        coordinator._run_order_reconciliation_cycle = AsyncMock()

        # Run for short time then stop
        task = asyncio.create_task(coordinator.run_order_reconciliation(interval_seconds=1))
        await asyncio.sleep(0.1)
        mock_bot.running = False
        await asyncio.sleep(0.1)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_run_order_reconciliation_cycle(self, coordinator, mock_bot):
        """Test single order reconciliation cycle"""
        mock_reconciler = Mock()
        mock_reconciler.fetch_local_open_orders = Mock(return_value={})
        mock_reconciler.fetch_exchange_open_orders = AsyncMock(return_value={})
        mock_reconciler.diff_orders = Mock(
            return_value=Mock(
                missing_on_exchange=[],
                missing_locally=[],
            )
        )
        mock_reconciler.record_snapshot = AsyncMock()

        await coordinator._run_order_reconciliation_cycle(mock_reconciler)

        mock_reconciler.fetch_local_open_orders.assert_called_once()
        mock_reconciler.fetch_exchange_open_orders.assert_called_once()
        mock_reconciler.diff_orders.assert_called_once()
        mock_reconciler.record_snapshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_decision_with_reduce_only_mode(
        self, coordinator, mock_bot, sample_decision, sample_product
    ):
        """Test decision execution in reduce-only mode"""
        mock_bot.is_reduce_only_mode = Mock(return_value=True)
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"

        with patch(
            "bot_v2.orchestration.execution_coordinator.OrderPlacementService"
        ) as mock_service_class:
            mock_service = AsyncMock()
            mock_service.execute_decision = AsyncMock(return_value=mock_order)
            mock_service_class.return_value = mock_service

            coordinator._order_placement_service = None

            await coordinator.execute_decision(
                symbol="BTC-USD",
                decision=sample_decision,
                mark=Decimal("50000"),
                product=sample_product,
                position_state=None,
            )

            call_kwargs = mock_service.execute_decision.call_args[1]
            assert call_kwargs["reduce_only_mode"] is True
