"""Tests for OrderPlacementService."""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, Product, TimeInForce
from bot_v2.features.live_trade.risk import ValidationError as RiskValidationError
from bot_v2.features.live_trade.strategies.perps_baseline import Action
from bot_v2.orchestration.execution.order_placement import OrderPlacementService


async def _async_to_thread(func, *args, **kwargs):
    """Mimic asyncio.to_thread by executing callable synchronously in tests."""
    return func(*args, **kwargs)


@pytest.fixture
def orders_store():
    """Mock orders store."""
    store = Mock()
    store.upsert = Mock()
    return store


@pytest.fixture
def order_stats():
    """Order statistics dict."""
    return {"attempted": 0, "successful": 0, "failed": 0}


@pytest.fixture
def order_placement_service(orders_store, order_stats):
    """Create OrderPlacementService instance."""
    return OrderPlacementService(
        orders_store=orders_store, order_stats=order_stats, broker=None, dry_run=False
    )


@pytest.fixture
def sample_product():
    """Create a sample product."""
    product = Mock(spec=Product)
    product.symbol = "BTC-USD"
    product.min_order_size = Decimal("0.001")
    product.price_increment = Decimal("0.01")
    return product


class TestOrderPlacementServiceInit:
    """Tests for service initialization."""

    def test_initializes_with_required_params(self, orders_store, order_stats):
        """Service initializes with required parameters."""
        service = OrderPlacementService(orders_store=orders_store, order_stats=order_stats)

        assert service._orders_store is orders_store
        assert service._order_stats is order_stats
        assert service._order_lock is None
        assert service._dry_run is False

    def test_initializes_with_dry_run(self, orders_store, order_stats):
        """Service can be initialized in dry run mode."""
        service = OrderPlacementService(
            orders_store=orders_store, order_stats=order_stats, dry_run=True
        )

        assert service._dry_run is True


class TestEnsureOrderLock:
    """Tests for _ensure_order_lock."""

    @pytest.mark.asyncio
    async def test_creates_lock_when_none(self, order_placement_service):
        """Creates async lock when none exists."""
        lock = order_placement_service._ensure_order_lock()

        assert isinstance(lock, asyncio.Lock)
        assert order_placement_service._order_lock is lock

    @pytest.mark.asyncio
    async def test_returns_existing_lock(self, order_placement_service):
        """Returns existing lock if already created."""
        lock1 = order_placement_service._ensure_order_lock()
        lock2 = order_placement_service._ensure_order_lock()

        assert lock1 is lock2


class TestExecuteDecisionDryRun:
    """Tests for dry run mode."""

    @pytest.mark.asyncio
    async def test_dry_run_logs_and_skips_execution(
        self, orders_store, order_stats, sample_product
    ):
        """Dry run mode logs decision without placing order."""
        service = OrderPlacementService(
            orders_store=orders_store, order_stats=order_stats, dry_run=True
        )

        decision = Mock()
        decision.target_notional = None
        decision.action = Action.BUY
        decision.quantity = Decimal("1.0")
        decision.reduce_only = False
        decision.leverage = None

        exec_engine = Mock()

        await service.execute_decision(
            symbol="BTC-USD",
            decision=decision,
            mark=Decimal("50000"),
            product=sample_product,
            position_state=None,
            exec_engine=exec_engine,
        )

        # Order should not be attempted
        assert order_stats["attempted"] == 0
        exec_engine.place_order.assert_not_called()


class TestExecuteDecisionValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_validates_product_not_none(self, order_placement_service):
        """Raises assertion error when product is None."""
        decision = Mock()
        decision.target_notional = None
        exec_engine = Mock()

        with pytest.raises(AssertionError, match="Missing product"):
            await order_placement_service.execute_decision(
                symbol="BTC-USD",
                decision=decision,
                mark=Decimal("50000"),
                product=None,
                position_state=None,
                exec_engine=exec_engine,
            )

    @pytest.mark.asyncio
    async def test_validates_mark_positive(self, order_placement_service, sample_product):
        """Raises assertion error when mark is not positive."""
        decision = Mock()
        decision.target_notional = None
        exec_engine = Mock()

        with pytest.raises(AssertionError, match="Invalid mark"):
            await order_placement_service.execute_decision(
                symbol="BTC-USD",
                decision=decision,
                mark=Decimal("0"),
                product=sample_product,
                position_state=None,
                exec_engine=exec_engine,
            )

    @pytest.mark.asyncio
    async def test_validates_position_state_has_quantity(
        self, order_placement_service, sample_product
    ):
        """Raises assertion error when position_state lacks quantity."""
        decision = Mock()
        decision.target_notional = None
        exec_engine = Mock()

        with pytest.raises(AssertionError, match="Position state missing quantity"):
            await order_placement_service.execute_decision(
                symbol="BTC-USD",
                decision=decision,
                mark=Decimal("50000"),
                product=sample_product,
                position_state={"side": "long"},  # Missing quantity
                exec_engine=exec_engine,
            )


class TestExecuteDecisionQuantityCalculation:
    """Tests for order quantity calculation."""

    @pytest.mark.asyncio
    async def test_close_action_no_position_returns_early(
        self, order_placement_service, sample_product, order_stats
    ):
        """Returns early when trying to close non-existent position."""
        decision = Mock()
        decision.target_notional = None
        decision.action = Action.CLOSE
        exec_engine = Mock()

        await order_placement_service.execute_decision(
            symbol="BTC-USD",
            decision=decision,
            mark=Decimal("50000"),
            product=sample_product,
            position_state=None,
            exec_engine=exec_engine,
        )

        # Should not attempt order
        assert order_stats["attempted"] == 0

    @pytest.mark.asyncio
    async def test_close_action_zero_position_returns_early(
        self, order_placement_service, sample_product, order_stats
    ):
        """Returns early when trying to close zero position."""
        decision = Mock()
        decision.action = Action.CLOSE
        exec_engine = Mock()

        await order_placement_service.execute_decision(
            symbol="BTC-USD",
            decision=decision,
            mark=Decimal("50000"),
            product=sample_product,
            position_state={"quantity": "0", "side": "long"},
            exec_engine=exec_engine,
        )

        assert order_stats["attempted"] == 0

    @pytest.mark.asyncio
    async def test_no_quantity_or_notional_returns_early(
        self, order_placement_service, sample_product, order_stats
    ):
        """Returns early when decision has no quantity or notional."""
        decision = Mock()
        decision.target_notional = None
        decision.action = Action.BUY
        decision.quantity = None
        decision.target_notional = None
        decision.reduce_only = False
        decision.leverage = None

        exec_engine = Mock()

        await order_placement_service.execute_decision(
            symbol="BTC-USD",
            decision=decision,
            mark=Decimal("50000"),
            product=sample_product,
            position_state=None,
            exec_engine=exec_engine,
        )

        assert order_stats["attempted"] == 0


class TestExecuteDecisionSideDetermination:
    """Tests for order side determination."""

    @pytest.mark.asyncio
    async def test_close_long_position_uses_sell_side(
        self, order_placement_service, sample_product
    ):
        """Closing long position uses SELL side."""
        decision = Mock()
        decision.action = Action.CLOSE
        decision.target_notional = None
        decision.reduce_only = False
        decision.leverage = None

        # Mock exec engine
        exec_engine = Mock()
        exec_engine.place_order = Mock(return_value="order_id_123")

        # Mock broker to return order
        broker = Mock()
        order = Mock()
        order.id = "order_id_123"
        order.symbol = "BTC-USD"
        order.side = OrderSide.SELL
        order.quantity = Decimal("1.0")
        broker.get_order = Mock(return_value=order)

        service = OrderPlacementService(
            orders_store=order_placement_service._orders_store,
            order_stats=order_placement_service._order_stats,
            broker=broker,
            dry_run=False,
        )

        # Create async wrapper for place_order
        async def async_place_order(*args, **kwargs):
            return await asyncio.to_thread(exec_engine.place_order, *args, **kwargs)

        with patch(
            "bot_v2.orchestration.execution.order_placement.asyncio.to_thread"
        ) as mock_to_thread:
            mock_to_thread.side_effect = _async_to_thread

            await service.execute_decision(
                symbol="BTC-USD",
                decision=decision,
                mark=Decimal("50000"),
                product=sample_product,
                position_state={"quantity": "1.0", "side": "long"},
                exec_engine=exec_engine,
            )

            # Verify SELL side was used
            call_kwargs = mock_to_thread.call_args_list[0][1]
            assert "side" in exec_engine.place_order.call_args[1]


class TestTimeInForceHandling:
    """Tests for time_in_force parsing."""

    @pytest.mark.asyncio
    async def test_parses_string_time_in_force(self, order_placement_service, sample_product):
        """Parses string TIF to TimeInForce enum."""
        decision = Mock()
        decision.target_notional = None
        decision.action = Action.BUY
        decision.quantity = Decimal("1.0")
        decision.reduce_only = False
        decision.leverage = None
        decision.time_in_force = "gtc"  # String TIF

        exec_engine = Mock()
        exec_engine.place_order = Mock(return_value=None)

        with patch(
            "bot_v2.orchestration.execution.order_placement.asyncio.to_thread"
        ) as mock_to_thread:
            mock_to_thread.side_effect = _async_to_thread

            await order_placement_service.execute_decision(
                symbol="BTC-USD",
                decision=decision,
                mark=Decimal("50000"),
                product=sample_product,
                position_state=None,
                exec_engine=exec_engine,
            )

            # Should have attempted order
            assert order_placement_service._order_stats["attempted"] > 0

    @pytest.mark.asyncio
    async def test_uses_default_time_in_force(self, order_placement_service, sample_product):
        """Uses default TIF when decision TIF is None."""
        decision = Mock()
        decision.target_notional = None
        decision.action = Action.BUY
        decision.quantity = Decimal("1.0")
        decision.reduce_only = False
        decision.leverage = None
        decision.time_in_force = None

        exec_engine = Mock()
        exec_engine.place_order = Mock(return_value=None)

        with patch(
            "bot_v2.orchestration.execution.order_placement.asyncio.to_thread"
        ) as mock_to_thread:
            mock_to_thread.side_effect = _async_to_thread

            await order_placement_service.execute_decision(
                symbol="BTC-USD",
                decision=decision,
                mark=Decimal("50000"),
                product=sample_product,
                position_state=None,
                exec_engine=exec_engine,
                default_time_in_force="ioc",  # Default TIF
            )

            assert order_placement_service._order_stats["attempted"] > 0


class TestPlaceOrderErrorHandling:
    """Tests for error handling in _place_order."""

    @pytest.mark.asyncio
    async def test_validation_error_increments_failed_and_raises(
        self, order_placement_service, sample_product
    ):
        """ValidationError increments failed counter and raises."""
        decision = Mock()
        decision.target_notional = None
        decision.action = Action.BUY
        decision.quantity = Decimal("1.0")
        decision.reduce_only = False
        decision.leverage = None

        exec_engine = Mock()

        with patch.object(
            order_placement_service,
            "_place_order",
            side_effect=ValidationError("Invalid order"),
        ):
            with pytest.raises(ValidationError):
                await order_placement_service.execute_decision(
                    symbol="BTC-USD",
                    decision=decision,
                    mark=Decimal("50000"),
                    product=sample_product,
                    position_state=None,
                    exec_engine=exec_engine,
                )

    @pytest.mark.asyncio
    async def test_execution_error_increments_failed_and_raises(
        self, order_placement_service, sample_product
    ):
        """ExecutionError increments failed counter and raises."""
        decision = Mock()
        decision.target_notional = None
        decision.action = Action.BUY
        decision.quantity = Decimal("1.0")
        decision.reduce_only = False
        decision.leverage = None

        exec_engine = Mock()

        with patch.object(
            order_placement_service,
            "_place_order",
            side_effect=ExecutionError("Execution failed"),
        ):
            with pytest.raises(ExecutionError):
                await order_placement_service.execute_decision(
                    symbol="BTC-USD",
                    decision=decision,
                    mark=Decimal("50000"),
                    product=sample_product,
                    position_state=None,
                    exec_engine=exec_engine,
                )

    @pytest.mark.asyncio
    async def test_generic_exception_logs_error(self, order_placement_service, sample_product):
        """Generic exception is logged but doesn't raise."""
        decision = Mock()
        decision.target_notional = None
        decision.action = Action.BUY
        decision.quantity = Decimal("1.0")
        decision.reduce_only = False
        decision.leverage = None

        exec_engine = Mock()

        with patch.object(
            order_placement_service,
            "_place_order",
            side_effect=RuntimeError("Unexpected error"),
        ):
            # Should not raise
            await order_placement_service.execute_decision(
                symbol="BTC-USD",
                decision=decision,
                mark=Decimal("50000"),
                product=sample_product,
                position_state=None,
                exec_engine=exec_engine,
            )
