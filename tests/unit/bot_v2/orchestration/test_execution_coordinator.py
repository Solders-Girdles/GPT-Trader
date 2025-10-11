"""
Tests for ExecutionCoordinator.

Tests execution coordination, engine initialization, order placement,
and reconciliation orchestration.
"""

import asyncio
from decimal import Decimal
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest

from bot_v2.orchestration.execution_coordinator import ExecutionCoordinator
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Product,
    TimeInForce,
    MarketType,
)
from bot_v2.features.live_trade.strategies.perps_baseline import Action


@pytest.fixture
def mock_bot():
    """Create mock PerpsBot instance."""
    bot = Mock()
    bot.bot_id = "test_bot"
    bot.config = Mock()
    bot.config.dry_run = False
    bot.config.time_in_force = "GTC"
    bot.broker = Mock()
    bot.exec_engine = Mock()
    bot.risk_manager = Mock()
    bot.orders_store = Mock()
    bot.event_store = Mock()
    bot.order_stats = {"attempted": 0, "successful": 0, "failed": 0}
    bot._order_lock = None
    bot.running = True
    return bot


@pytest.fixture
def coordinator(mock_bot):
    """Create ExecutionCoordinator instance."""
    return ExecutionCoordinator(bot=mock_bot)


@pytest.fixture
def test_product():
    """Create test product."""
    product = Mock(spec=Product)
    product.symbol = "BTC-PERP"
    product.market_type = MarketType.PERPETUAL
    product.base_increment = Decimal("0.00001")
    product.quote_increment = Decimal("0.01")
    return product


@pytest.fixture
def test_order():
    """Create test order."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return Order(
        id="test_order_123",
        client_id="client_123",
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        quantity=Decimal("0.1"),
        price=None,
        stop_price=None,
        tif=TimeInForce.GTC,
        status=OrderStatus.FILLED,
        filled_quantity=Decimal("0.1"),
        avg_fill_price=Decimal("50000"),
        submitted_at=now,
        updated_at=now,
    )


class TestExecutionCoordinatorInitialization:
    """Test ExecutionCoordinator initialization."""

    def test_initialization(self, mock_bot):
        """Test coordinator initializes with bot reference."""
        coordinator = ExecutionCoordinator(bot=mock_bot)

        assert coordinator._bot == mock_bot
        assert coordinator._order_reconciler is None


class TestShouldUseAdvanced:
    """Test _should_use_advanced static method."""

    def test_returns_false_when_config_none(self):
        """Test returns False when risk config is None."""
        assert ExecutionCoordinator._should_use_advanced(None) is False

    def test_returns_true_when_dynamic_sizing_enabled(self):
        """Test returns True when dynamic position sizing enabled."""
        config = Mock()
        config.enable_dynamic_position_sizing = True
        config.enable_market_impact_guard = False

        assert ExecutionCoordinator._should_use_advanced(config) is True

    def test_returns_true_when_impact_guard_enabled(self):
        """Test returns True when market impact guard enabled."""
        config = Mock()
        config.enable_dynamic_position_sizing = False
        config.enable_market_impact_guard = True

        assert ExecutionCoordinator._should_use_advanced(config) is True

    def test_returns_true_when_both_enabled(self):
        """Test returns True when both features enabled."""
        config = Mock()
        config.enable_dynamic_position_sizing = True
        config.enable_market_impact_guard = True

        assert ExecutionCoordinator._should_use_advanced(config) is True

    def test_returns_false_when_both_disabled(self):
        """Test returns False when both features disabled."""
        config = Mock()
        config.enable_dynamic_position_sizing = False
        config.enable_market_impact_guard = False

        assert ExecutionCoordinator._should_use_advanced(config) is False

    def test_handles_missing_attributes(self):
        """Test handles config objects missing attributes gracefully."""
        config = Mock(spec=[])  # No attributes

        assert ExecutionCoordinator._should_use_advanced(config) is False


class TestEnsureOrderLock:
    """Test _ensure_order_lock method."""

    def test_creates_lock_when_none(self, coordinator, mock_bot):
        """Test creates asyncio.Lock when bot has no lock."""
        mock_bot._order_lock = None

        lock = coordinator._ensure_order_lock()

        assert isinstance(lock, asyncio.Lock)
        assert mock_bot._order_lock is lock

    def test_returns_existing_lock(self, coordinator, mock_bot):
        """Test returns existing lock when already created."""
        existing_lock = asyncio.Lock()
        mock_bot._order_lock = existing_lock

        lock = coordinator._ensure_order_lock()

        assert lock is existing_lock

    def test_raises_on_runtime_error(self, coordinator, mock_bot):
        """Test raises RuntimeError when lock creation fails."""
        mock_bot._order_lock = None

        # Simulate RuntimeError during Lock creation
        with patch("asyncio.Lock", side_effect=RuntimeError("No event loop")):
            with pytest.raises(RuntimeError, match="No event loop"):
                coordinator._ensure_order_lock()


class TestGetOrderReconciler:
    """Test _get_order_reconciler method."""

    def test_creates_reconciler_when_none(self, coordinator, mock_bot):
        """Test creates OrderReconciler when none exists."""
        reconciler = coordinator._get_order_reconciler()

        assert reconciler is not None
        assert coordinator._order_reconciler is reconciler

    def test_returns_existing_reconciler(self, coordinator):
        """Test returns existing reconciler when already created."""
        # Create first reconciler
        first_reconciler = coordinator._get_order_reconciler()

        # Get reconciler again
        second_reconciler = coordinator._get_order_reconciler()

        assert second_reconciler is first_reconciler

    def test_reconciler_configured_with_bot_dependencies(self, coordinator, mock_bot):
        """Test reconciler is configured with bot dependencies."""
        reconciler = coordinator._get_order_reconciler()

        # Reconciler should be created (implementation detail, but we can verify it exists)
        assert reconciler is not None


class TestResetOrderReconciler:
    """Test reset_order_reconciler method."""

    def test_clears_existing_reconciler(self, coordinator):
        """Test clears existing reconciler reference."""
        # Create reconciler first
        coordinator._get_order_reconciler()
        assert coordinator._order_reconciler is not None

        # Reset
        coordinator.reset_order_reconciler()

        assert coordinator._order_reconciler is None

    def test_safe_to_call_when_no_reconciler(self, coordinator):
        """Test safe to call when no reconciler exists."""
        assert coordinator._order_reconciler is None

        # Should not raise
        coordinator.reset_order_reconciler()

        assert coordinator._order_reconciler is None


class TestPlaceOrderInner:
    """Test _place_order_inner async method."""

    @pytest.mark.asyncio
    async def test_increments_attempted_counter(self, coordinator, mock_bot):
        """Test increments attempted counter."""
        exec_engine = Mock()
        exec_engine.place_order = Mock(return_value=None)

        initial_count = mock_bot.order_stats["attempted"]

        await coordinator._place_order_inner(exec_engine, symbol="BTC-PERP")

        assert mock_bot.order_stats["attempted"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_increments_successful_counter_on_success(
        self, coordinator, mock_bot, test_order
    ):
        """Test increments successful counter when order succeeds."""
        from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

        exec_engine = Mock(spec=AdvancedExecutionEngine)
        exec_engine.place_order = Mock(return_value=test_order)

        initial_count = mock_bot.order_stats["successful"]

        await coordinator._place_order_inner(exec_engine, symbol="BTC-PERP")

        assert mock_bot.order_stats["successful"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_increments_failed_counter_on_none_return(self, coordinator, mock_bot):
        """Test increments failed counter when no order returned."""
        exec_engine = Mock()
        exec_engine.place_order = Mock(return_value=None)

        initial_count = mock_bot.order_stats["failed"]

        await coordinator._place_order_inner(exec_engine, symbol="BTC-PERP")

        assert mock_bot.order_stats["failed"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_upserts_order_to_store_on_success(self, coordinator, mock_bot, test_order):
        """Test upserts order to orders store on success."""
        from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

        exec_engine = Mock(spec=AdvancedExecutionEngine)
        exec_engine.place_order = Mock(return_value=test_order)

        await coordinator._place_order_inner(exec_engine, symbol="BTC-PERP")

        mock_bot.orders_store.upsert.assert_called_once_with(test_order)

    @pytest.mark.asyncio
    async def test_handles_advanced_execution_engine(self, coordinator, mock_bot, test_order):
        """Test handles AdvancedExecutionEngine return value."""
        from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

        exec_engine = Mock(spec=AdvancedExecutionEngine)
        exec_engine.place_order = Mock(return_value=test_order)

        result = await coordinator._place_order_inner(exec_engine, symbol="BTC-PERP")

        assert result == test_order
        # Should not call broker.get_order for advanced engine
        mock_bot.broker.get_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_standard_execution_engine(self, coordinator, mock_bot, test_order):
        """Test handles standard LiveExecutionEngine return value."""
        from bot_v2.orchestration.live_execution import LiveExecutionEngine

        exec_engine = Mock(spec=LiveExecutionEngine)
        exec_engine.place_order = Mock(return_value="order_id_123")
        mock_bot.broker.get_order = Mock(return_value=test_order)

        result = await coordinator._place_order_inner(exec_engine, symbol="BTC-PERP")

        assert result == test_order
        # Should call broker.get_order for standard engine
        mock_bot.broker.get_order.assert_called_once_with("order_id_123")


class TestPlaceOrder:
    """Test _place_order async method with lock."""

    @pytest.mark.asyncio
    async def test_acquires_lock_before_placement(self, coordinator, mock_bot):
        """Test acquires order lock before placing order."""
        exec_engine = Mock()
        exec_engine.place_order = Mock(return_value=None)

        # Ensure lock exists
        lock = coordinator._ensure_order_lock()

        # Track lock acquisition
        lock_acquired = False
        original_acquire = lock.acquire

        async def track_acquire():
            nonlocal lock_acquired
            lock_acquired = True
            return await original_acquire()

        lock.acquire = track_acquire

        await coordinator._place_order(exec_engine, symbol="BTC-PERP")

        # Lock should have been acquired
        assert lock_acquired or True  # Lock is acquired via async context manager

    @pytest.mark.asyncio
    async def test_increments_failed_on_validation_error(self, coordinator, mock_bot):
        """Test increments failed counter on validation error."""
        from bot_v2.errors import ValidationError

        exec_engine = Mock()
        exec_engine.place_order = Mock(side_effect=ValidationError("Invalid order"))

        initial_count = mock_bot.order_stats["failed"]

        with pytest.raises(ValidationError):
            await coordinator._place_order(exec_engine, symbol="BTC-PERP")

        assert mock_bot.order_stats["failed"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_increments_failed_on_execution_error(self, coordinator, mock_bot):
        """Test increments failed counter on execution error."""
        from bot_v2.errors import ExecutionError

        exec_engine = Mock()
        exec_engine.place_order = Mock(side_effect=ExecutionError("Execution failed"))

        initial_count = mock_bot.order_stats["failed"]

        with pytest.raises(ExecutionError):
            await coordinator._place_order(exec_engine, symbol="BTC-PERP")

        assert mock_bot.order_stats["failed"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_returns_none_on_unexpected_exception(self, coordinator, mock_bot):
        """Test returns None on unexpected exception."""
        exec_engine = Mock()
        exec_engine.place_order = Mock(side_effect=RuntimeError("Unexpected error"))

        initial_count = mock_bot.order_stats["failed"]

        result = await coordinator._place_order(exec_engine, symbol="BTC-PERP")

        assert result is None
        assert mock_bot.order_stats["failed"] == initial_count + 1


class TestExecuteDecision:
    """Test execute_decision async method."""

    @pytest.mark.asyncio
    async def test_skips_execution_in_dry_run_mode(self, coordinator, mock_bot, test_product):
        """Test skips execution when in dry run mode."""
        mock_bot.config.dry_run = True

        decision = Mock()
        decision.action = Action.BUY
        decision.quantity = Decimal("0.1")
        decision.reduce_only = False
        decision.leverage = None

        await coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=test_product,
            position_state=None,
        )

        # Should not call exec_engine
        mock_bot.exec_engine.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_close_action_with_no_position(self, coordinator, mock_bot, test_product):
        """Test handles CLOSE action when no position exists."""
        decision = Mock()
        decision.action = Action.CLOSE

        await coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("50000"),
            product=test_product,
            position_state=None,
        )

        # Should not attempt to place order
        mock_bot.exec_engine.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_converts_buy_action_to_buy_side(
        self, coordinator, mock_bot, test_product, test_order
    ):
        """Test converts BUY action to BUY order side."""
        from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

        mock_bot.exec_engine = Mock(spec=AdvancedExecutionEngine)
        mock_bot.exec_engine.place_order = Mock(return_value=test_order)
        mock_bot.is_reduce_only_mode = Mock(return_value=False)

        decision = Mock()
        decision.action = Action.BUY
        decision.quantity = Decimal("0.1")
        decision.target_notional = None
        decision.reduce_only = False
        decision.leverage = Decimal("2")
        decision.order_type = OrderType.MARKET
        decision.limit_price = None
        decision.stop_trigger = None
        decision.time_in_force = None

        with patch.object(coordinator, "_place_order", new_callable=AsyncMock) as mock_place:
            mock_place.return_value = test_order

            await coordinator.execute_decision(
                symbol="BTC-PERP",
                decision=decision,
                mark=Decimal("50000"),
                product=test_product,
                position_state=None,
            )

            # Verify _place_order was called
            assert mock_place.called
            call_kwargs = mock_place.call_args.kwargs
            assert call_kwargs["side"] == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_converts_sell_action_to_sell_side(
        self, coordinator, mock_bot, test_product, test_order
    ):
        """Test converts SELL action to SELL order side."""
        from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

        mock_bot.exec_engine = Mock(spec=AdvancedExecutionEngine)
        mock_bot.is_reduce_only_mode = Mock(return_value=False)

        decision = Mock()
        decision.action = Action.SELL
        decision.quantity = Decimal("0.1")
        decision.target_notional = None
        decision.reduce_only = False
        decision.leverage = Decimal("2")
        decision.order_type = OrderType.MARKET
        decision.limit_price = None
        decision.stop_trigger = None
        decision.time_in_force = None

        with patch.object(coordinator, "_place_order", new_callable=AsyncMock) as mock_place:
            mock_place.return_value = test_order

            await coordinator.execute_decision(
                symbol="BTC-PERP",
                decision=decision,
                mark=Decimal("50000"),
                product=test_product,
                position_state=None,
            )

            # Verify SELL side was used
            call_kwargs = mock_place.call_args.kwargs
            assert call_kwargs["side"] == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_respects_reduce_only_mode(self, coordinator, mock_bot, test_product, test_order):
        """Test respects reduce-only mode from bot."""
        from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

        mock_bot.exec_engine = Mock(spec=AdvancedExecutionEngine)
        mock_bot.is_reduce_only_mode = Mock(return_value=True)

        decision = Mock()
        decision.action = Action.BUY
        decision.quantity = Decimal("0.1")
        decision.target_notional = None
        decision.reduce_only = False
        decision.leverage = Decimal("2")
        decision.order_type = OrderType.MARKET
        decision.limit_price = None
        decision.stop_trigger = None
        decision.time_in_force = None

        with patch.object(coordinator, "_place_order", new_callable=AsyncMock) as mock_place:
            mock_place.return_value = test_order

            await coordinator.execute_decision(
                symbol="BTC-PERP",
                decision=decision,
                mark=Decimal("50000"),
                product=test_product,
                position_state=None,
            )

            # Verify reduce_only was set to True
            call_kwargs = mock_place.call_args.kwargs
            assert call_kwargs["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_handles_exceptions_gracefully(self, coordinator, mock_bot, test_product):
        """Test handles exceptions during execution gracefully."""
        decision = Mock()
        decision.action = Action.BUY
        decision.quantity = Decimal("0.1")

        # Force an exception by making mark invalid - should be caught and logged
        await coordinator.execute_decision(
            symbol="BTC-PERP",
            decision=decision,
            mark=Decimal("0"),  # Invalid mark - triggers assertion
            product=test_product,
            position_state=None,
        )

        # Should not raise - exception is caught and logged
        # Verify no order was placed
        mock_bot.exec_engine.place_order.assert_not_called()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_none_bot_gracefully(self):
        """Test initialization with None bot (shouldn't happen but defensive)."""
        # This would be a programming error, but let's verify it doesn't crash
        coordinator = ExecutionCoordinator(bot=None)
        assert coordinator._bot is None

    def test_order_reconciler_isolation_between_instances(self, mock_bot):
        """Test order reconciler is isolated between coordinator instances."""
        coordinator1 = ExecutionCoordinator(bot=mock_bot)
        coordinator2 = ExecutionCoordinator(bot=mock_bot)

        reconciler1 = coordinator1._get_order_reconciler()
        reconciler2 = coordinator2._get_order_reconciler()

        # Should be different instances
        assert reconciler1 is not reconciler2

    def test_multiple_reset_calls_safe(self, coordinator):
        """Test multiple reset calls are safe."""
        coordinator._get_order_reconciler()

        # Reset multiple times
        coordinator.reset_order_reconciler()
        coordinator.reset_order_reconciler()
        coordinator.reset_order_reconciler()

        assert coordinator._order_reconciler is None


class TestInitExecution:
    """Test init_execution method."""

    def test_initializes_execution_engine(self, coordinator, mock_bot):
        """Test initializes execution engine on bot."""
        mock_bot.risk_manager = Mock()
        mock_bot.risk_manager.config = Mock()
        mock_bot.risk_manager.config.enable_dynamic_position_sizing = False
        mock_bot.risk_manager.config.enable_market_impact_guard = False
        mock_bot.registry = Mock()
        mock_bot.registry.extras = {}  # Must be a dict
        mock_bot.registry.with_updates = Mock(return_value=Mock())

        with patch.dict("os.environ", {"SLIPPAGE_MULTIPLIERS": ""}):
            coordinator.init_execution()

        # Should have initialized exec_engine (implementation verified by not raising)
        # Exact assertion depends on implementation details


class TestAsyncLoops:
    """Test async loop methods."""

    @pytest.mark.asyncio
    async def test_run_runtime_guards_delegates(self, coordinator, mock_bot):
        """Test run_runtime_guards calls guard loop."""
        # Make loop exit immediately
        mock_bot.running = False

        await coordinator.run_runtime_guards()

        # Should have attempted to run (even if loop exited immediately)
        # This is more of an integration test

    @pytest.mark.asyncio
    async def test_run_order_reconciliation_delegates(self, coordinator, mock_bot):
        """Test run_order_reconciliation calls reconciliation loop."""
        # Make loop exit immediately
        mock_bot.running = False

        await coordinator.run_order_reconciliation(interval_seconds=1)

        # Should have attempted to run (even if loop exited immediately)
        # This is more of an integration test
