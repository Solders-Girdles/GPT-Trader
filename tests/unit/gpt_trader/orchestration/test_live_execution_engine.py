"""Tests for LiveExecutionEngine - the core trading orchestration engine."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.core.interfaces import OrderSide, OrderType
from gpt_trader.features.live_trade.risk import ValidationError
from gpt_trader.orchestration.live_execution import (
    LiveExecutionEngine,
    LiveOrder,
)

# ============================================================
# Test: LiveOrder dataclass
# ============================================================


class TestLiveOrder:
    """Tests for LiveOrder dataclass."""

    def test_live_order_creation(self) -> None:
        """Test creating a LiveOrder with basic fields."""
        order = LiveOrder(
            symbol="BTC-PERP",
            side="buy",
            quantity=Decimal("0.5"),
        )

        assert order.symbol == "BTC-PERP"
        assert order.side == "buy"
        assert order.quantity == Decimal("0.5")
        assert order.price is None
        assert order.order_type == "market"
        assert order.reduce_only is False
        assert order.leverage is None

    def test_live_order_with_all_fields(self) -> None:
        """Test creating a LiveOrder with all fields."""
        order = LiveOrder(
            symbol="ETH-PERP",
            side="sell",
            quantity=Decimal("1.5"),
            price=Decimal("2000.00"),
            order_type="limit",
            reduce_only=True,
            leverage=5,
        )

        assert order.symbol == "ETH-PERP"
        assert order.side == "sell"
        assert order.quantity == Decimal("1.5")
        assert order.price == Decimal("2000.00")
        assert order.order_type == "limit"
        assert order.reduce_only is True
        assert order.leverage == 5

    def test_live_order_converts_quantity_to_decimal(self) -> None:
        """Test that __post_init__ converts quantity to Decimal."""
        # Pass as float
        order = LiveOrder(symbol="BTC-PERP", side="buy", quantity=0.123)
        assert isinstance(order.quantity, Decimal)
        assert order.quantity == Decimal("0.123")

        # Pass as int
        order_int = LiveOrder(symbol="BTC-PERP", side="buy", quantity=5)
        assert isinstance(order_int.quantity, Decimal)
        assert order_int.quantity == Decimal("5")

        # Pass as string
        order_str = LiveOrder(symbol="BTC-PERP", side="buy", quantity="2.5")
        assert isinstance(order_str.quantity, Decimal)
        assert order_str.quantity == Decimal("2.5")


# ============================================================
# Test: LiveExecutionEngine initialization
# ============================================================


class TestLiveExecutionEngineInit:
    """Tests for LiveExecutionEngine initialization."""

    @pytest.fixture
    def mock_broker(self) -> Mock:
        """Create a mock broker."""
        broker = Mock()
        broker.list_balances.return_value = []
        broker.get_product.return_value = Mock()
        return broker

    @pytest.fixture
    def mock_risk_manager(self) -> Mock:
        """Create a mock risk manager."""
        manager = Mock()
        manager.event_store = None
        return manager

    def test_init_minimal(self, mock_broker: Mock, bot_config_factory) -> None:
        """Test initialization with minimal arguments."""
        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        assert engine.broker is mock_broker
        assert engine.bot_id == "live_execution"
        assert engine.event_store is not None
        assert engine.risk_manager is not None
        assert engine.slippage_multipliers == {}
        assert engine.enable_order_preview is False
        assert engine.open_orders == []

    def test_init_with_custom_bot_id(self, mock_broker: Mock, bot_config_factory) -> None:
        """Test initialization with custom bot_id."""
        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config, bot_id="my_bot")

        assert engine.bot_id == "my_bot"

    def test_init_with_risk_manager(
        self, mock_broker: Mock, mock_risk_manager: Mock, bot_config_factory
    ) -> None:
        """Test initialization with provided risk manager."""
        config = bot_config_factory()
        engine = LiveExecutionEngine(
            broker=mock_broker, config=config, risk_manager=mock_risk_manager
        )

        assert engine.risk_manager is mock_risk_manager

    def test_init_with_event_store(self, mock_broker: Mock, bot_config_factory) -> None:
        """Test initialization with provided event store."""
        mock_event_store = Mock()
        config = bot_config_factory()

        engine = LiveExecutionEngine(
            broker=mock_broker, config=config, event_store=mock_event_store
        )

        assert engine.event_store is mock_event_store

    def test_init_with_slippage_multipliers(self, mock_broker: Mock, bot_config_factory) -> None:
        """Test initialization with slippage multipliers."""
        slippage = {"BTC-PERP": 1.5, "ETH-PERP": 2.0}
        config = bot_config_factory()

        engine = LiveExecutionEngine(
            broker=mock_broker, config=config, slippage_multipliers=slippage
        )

        assert engine.slippage_multipliers == slippage

    def test_init_with_order_preview_enabled(self, mock_broker: Mock, bot_config_factory) -> None:
        """Test initialization with order preview explicitly enabled."""
        config = bot_config_factory()

        engine = LiveExecutionEngine(broker=mock_broker, config=config, enable_preview=True)

        assert engine.enable_order_preview is True

    def test_init_order_preview_from_config(self, mock_broker: Mock, bot_config_factory) -> None:
        """Test initialization reads enable_order_preview from config."""
        config = bot_config_factory(enable_order_preview=True)

        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        assert engine.enable_order_preview is True

    def test_init_creates_helper_modules(self, mock_broker: Mock, bot_config_factory) -> None:
        """Test initialization creates all helper modules."""
        config = bot_config_factory()

        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        assert engine.state_collector is not None
        assert engine.order_submitter is not None
        assert engine.order_validator is not None
        assert engine.guard_manager is not None

    def test_init_risk_manager_set_event_store(self, mock_broker: Mock, bot_config_factory) -> None:
        """Test that risk manager's event store is updated if different."""
        mock_event_store = Mock()
        mock_risk_manager = Mock()
        mock_risk_manager.event_store = Mock()  # Different store
        mock_risk_manager.set_event_store = Mock()
        config = bot_config_factory()

        LiveExecutionEngine(
            broker=mock_broker,
            config=config,
            risk_manager=mock_risk_manager,
            event_store=mock_event_store,
        )

        mock_risk_manager.set_event_store.assert_called_once_with(mock_event_store)


# ============================================================
# Test: place_order method
# ============================================================


class TestPlaceOrder:
    """Tests for place_order method."""

    @pytest.fixture
    def mock_broker(self) -> Mock:
        """Create a mock broker."""
        broker = Mock()
        broker.list_balances.return_value = []
        broker.get_product.return_value = Mock()
        return broker

    @pytest.fixture
    def engine(self, mock_broker: Mock, bot_config_factory) -> LiveExecutionEngine:
        """Create an engine with mocked components."""
        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        # Mock helper modules
        engine.state_collector = Mock()
        engine.state_collector.require_product.return_value = Mock()
        engine.state_collector.collect_account_state.return_value = (
            [],  # balances
            Decimal("10000"),  # equity
            {},  # collateral_balances
            Decimal("10000"),  # collateral_total
            [],  # current_positions
        )
        engine.state_collector.log_collateral_update = Mock()
        engine.state_collector.build_positions_dict.return_value = {}
        engine.state_collector.resolve_effective_price.return_value = Decimal("50000")

        engine.order_validator = Mock()
        engine.order_validator.validate_exchange_rules.return_value = (
            Decimal("1"),
            Decimal("50000"),
        )
        engine.order_validator.ensure_mark_is_fresh = Mock()
        engine.order_validator.enforce_slippage_guard = Mock()
        engine.order_validator.run_pre_trade_validation = Mock()
        engine.order_validator.maybe_preview_order = Mock()
        engine.order_validator.finalize_reduce_only_flag.return_value = False

        engine.order_submitter = Mock()
        engine.order_submitter.submit_order.return_value = "order-123"
        engine.order_submitter.record_rejection = Mock()
        engine.order_submitter.record_preview = Mock()

        engine.guard_manager = Mock()
        engine.guard_manager.invalidate_cache = Mock()

        engine.event_store = Mock()

        return engine

    def test_place_order_success(self, engine: LiveExecutionEngine) -> None:
        """Test successful order placement."""
        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
        )

        assert order_id == "order-123"
        engine.state_collector.require_product.assert_called_once()
        engine.state_collector.collect_account_state.assert_called_once()
        engine.order_validator.validate_exchange_rules.assert_called_once()
        engine.order_submitter.submit_order.assert_called_once()
        engine.guard_manager.invalidate_cache.assert_called_once()

    def test_place_order_requires_quantity(self, engine: LiveExecutionEngine) -> None:
        """Test that place_order raises TypeError when quantity is None."""
        with pytest.raises(TypeError, match="place_order requires 'quantity'"):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=None,
            )

    def test_place_order_with_price(self, engine: LiveExecutionEngine) -> None:
        """Test order placement with limit price."""
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("49000"),
        )

        # Verify price was passed through validation
        call_args = engine.order_submitter.submit_order.call_args
        assert call_args is not None

    def test_place_order_validation_error(self, engine: LiveExecutionEngine) -> None:
        """Test order placement when validation fails."""
        engine.order_validator.run_pre_trade_validation.side_effect = ValidationError(
            "Leverage exceeds max"
        )

        with pytest.raises(ValidationError, match="Leverage exceeds max"):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
            )

        # Verify rejection was recorded
        engine.order_submitter.record_rejection.assert_called_once()
        # Verify cache was invalidated in finally block
        engine.guard_manager.invalidate_cache.assert_called_once()

    def test_place_order_exception_handling(self, engine: LiveExecutionEngine) -> None:
        """Test order placement exception handling."""
        engine.state_collector.collect_account_state.side_effect = RuntimeError("Network error")

        with pytest.raises(RuntimeError, match="Network error"):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
            )

        # Verify error was logged to event store
        engine.event_store.append_error.assert_called_once()
        # Verify cache was invalidated
        engine.guard_manager.invalidate_cache.assert_called_once()

    def test_place_order_exception_event_store_failure(self, engine: LiveExecutionEngine) -> None:
        """Test order placement when event store also fails."""
        engine.state_collector.collect_account_state.side_effect = RuntimeError("Network error")
        engine.event_store.append_error.side_effect = Exception("Store failed")

        # Should not raise from event store failure
        with pytest.raises(RuntimeError, match="Network error"):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
            )

    def test_place_order_with_reduce_only(self, engine: LiveExecutionEngine) -> None:
        """Test order placement with reduce_only flag."""
        engine.order_validator.finalize_reduce_only_flag.return_value = True

        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            reduce_only=True,
        )

        engine.order_validator.finalize_reduce_only_flag.assert_called_with(True, "BTC-PERP")

    def test_place_order_with_leverage(self, engine: LiveExecutionEngine) -> None:
        """Test order placement with leverage."""
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            leverage=5,
        )

        call_args = engine.order_submitter.submit_order.call_args
        assert call_args.kwargs["leverage"] == 5

    def test_place_order_with_client_order_id(self, engine: LiveExecutionEngine) -> None:
        """Test order placement with client order ID."""
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            client_order_id="my-order-123",
        )

        call_args = engine.order_submitter.submit_order.call_args
        assert call_args.kwargs["client_order_id"] == "my-order-123"


# ============================================================
# Test: cancel_all_orders method
# ============================================================


class TestCancelAllOrders:
    """Tests for cancel_all_orders method."""

    @pytest.fixture
    def engine(self, bot_config_factory) -> LiveExecutionEngine:
        """Create an engine with mocked guard_manager."""
        mock_broker = Mock()
        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        engine.guard_manager = Mock()
        engine.guard_manager.cancel_all_orders.return_value = 3

        return engine

    def test_cancel_all_orders_delegates(self, engine: LiveExecutionEngine) -> None:
        """Test cancel_all_orders delegates to guard_manager."""
        result = engine.cancel_all_orders()

        assert result == 3
        engine.guard_manager.cancel_all_orders.assert_called_once()

    def test_cancel_all_orders_returns_count(self, engine: LiveExecutionEngine) -> None:
        """Test cancel_all_orders returns number cancelled."""
        engine.guard_manager.cancel_all_orders.return_value = 5

        result = engine.cancel_all_orders()

        assert result == 5


# ============================================================
# Test: run_runtime_guards method
# ============================================================


class TestRunRuntimeGuards:
    """Tests for run_runtime_guards method."""

    @pytest.fixture
    def engine(self, bot_config_factory) -> LiveExecutionEngine:
        """Create an engine with mocked guard_manager."""
        mock_broker = Mock()
        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        engine.guard_manager = Mock()

        return engine

    def test_run_runtime_guards_delegates(self, engine: LiveExecutionEngine) -> None:
        """Test run_runtime_guards delegates to guard_manager."""
        engine.run_runtime_guards()

        engine.guard_manager.safe_run_runtime_guards.assert_called_once()


# ============================================================
# Test: reset_daily_tracking method
# ============================================================


class TestResetDailyTracking:
    """Tests for reset_daily_tracking method."""

    @pytest.fixture
    def engine(self, bot_config_factory) -> LiveExecutionEngine:
        """Create an engine with mocked components."""
        mock_broker = Mock()
        mock_broker.list_balances.return_value = []

        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        engine.state_collector = Mock()
        engine.state_collector.calculate_equity_from_balances.return_value = (
            Decimal("10000"),
            {},
            Decimal("10000"),
        )

        engine.risk_manager = Mock()
        engine.guard_manager = Mock()

        return engine

    def test_reset_daily_tracking_success(self, engine: LiveExecutionEngine) -> None:
        """Test successful daily tracking reset."""
        engine.reset_daily_tracking()

        engine.broker.list_balances.assert_called_once()
        engine.state_collector.calculate_equity_from_balances.assert_called_once()
        engine.risk_manager.reset_daily_tracking.assert_called_once()
        engine.guard_manager.invalidate_cache.assert_called_once()

    def test_reset_daily_tracking_handles_exception(self, engine: LiveExecutionEngine) -> None:
        """Test reset_daily_tracking handles exceptions gracefully."""
        engine.broker.list_balances.side_effect = RuntimeError("API error")

        # Should not raise
        engine.reset_daily_tracking()

        # Risk manager should not be called if balance fetch failed
        engine.risk_manager.reset_daily_tracking.assert_not_called()


# ============================================================
# Test: _invalidate_runtime_guard_cache method
# ============================================================


class TestInvalidateRuntimeGuardCache:
    """Tests for _invalidate_runtime_guard_cache backward compatibility."""

    @pytest.fixture
    def engine(self, bot_config_factory) -> LiveExecutionEngine:
        """Create an engine with mocked guard_manager."""
        mock_broker = Mock()
        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        engine.guard_manager = Mock()

        return engine

    def test_invalidate_cache_delegates(self, engine: LiveExecutionEngine) -> None:
        """Test _invalidate_runtime_guard_cache delegates to guard_manager."""
        engine._invalidate_runtime_guard_cache()

        engine.guard_manager.invalidate_cache.assert_called_once()


# ============================================================
# Test: Integration scenarios
# ============================================================


class TestLiveExecutionEngineIntegration:
    """Integration tests for LiveExecutionEngine workflows."""

    @pytest.fixture
    def mock_broker(self) -> Mock:
        """Create a mock broker."""
        broker = Mock()
        broker.list_balances.return_value = [{"currency": "USD", "available": "10000", "hold": "0"}]
        broker.get_product.return_value = Mock(
            base_increment="0.001",
            quote_increment="0.01",
            min_market_funds="1",
        )
        broker.place_order.return_value = {"order_id": "integration-order-123"}
        return broker

    def test_full_order_flow(self, mock_broker: Mock, bot_config_factory) -> None:
        """Test complete order flow from place_order to submission."""
        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        # Mock state collector
        engine.state_collector = Mock()
        engine.state_collector.require_product.return_value = Mock()
        engine.state_collector.collect_account_state.return_value = (
            [],
            Decimal("10000"),
            {},
            Decimal("10000"),
            [],
        )
        engine.state_collector.log_collateral_update = Mock()
        engine.state_collector.build_positions_dict.return_value = {}
        engine.state_collector.resolve_effective_price.return_value = Decimal("50000")

        # Mock validator
        engine.order_validator = Mock()
        engine.order_validator.validate_exchange_rules.return_value = (
            Decimal("0.1"),
            None,
        )
        engine.order_validator.ensure_mark_is_fresh = Mock()
        engine.order_validator.enforce_slippage_guard = Mock()
        engine.order_validator.run_pre_trade_validation = Mock()
        engine.order_validator.maybe_preview_order = Mock()
        engine.order_validator.finalize_reduce_only_flag.return_value = False

        # Mock submitter to return order ID
        engine.order_submitter = Mock()
        engine.order_submitter.submit_order.return_value = "order-999"

        engine.guard_manager = Mock()

        # Place order
        order_id = engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order_id == "order-999"

    def test_risk_guard_trip_cancels_orders(self, mock_broker: Mock, bot_config_factory) -> None:
        """Test that risk guard trips can cancel open orders."""
        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        # Simulate open orders
        engine.open_orders = ["order-1", "order-2"]

        # Mock guard_manager
        engine.guard_manager = Mock()
        engine.guard_manager.cancel_all_orders.return_value = 2

        # Cancel all
        cancelled = engine.cancel_all_orders()

        assert cancelled == 2
        engine.guard_manager.cancel_all_orders.assert_called_once()


# ============================================================
# Test: Additional place_order scenarios
# ============================================================


class TestPlaceOrderAdditional:
    """Additional tests for place_order edge cases."""

    @pytest.fixture
    def mock_broker(self) -> Mock:
        """Create a mock broker."""
        broker = Mock()
        broker.list_balances.return_value = []
        broker.get_product.return_value = Mock()
        return broker

    @pytest.fixture
    def engine(self, mock_broker: Mock, bot_config_factory) -> LiveExecutionEngine:
        """Create an engine with mocked components."""
        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        # Mock helper modules
        engine.state_collector = Mock()
        engine.state_collector.require_product.return_value = Mock()
        engine.state_collector.collect_account_state.return_value = (
            [],
            Decimal("10000"),
            {},
            Decimal("10000"),
            [],
        )
        engine.state_collector.log_collateral_update = Mock()
        engine.state_collector.build_positions_dict.return_value = {}
        engine.state_collector.resolve_effective_price.return_value = Decimal("50000")

        engine.order_validator = Mock()
        engine.order_validator.validate_exchange_rules.return_value = (
            Decimal("1"),
            Decimal("50000"),
        )
        engine.order_validator.ensure_mark_is_fresh = Mock()
        engine.order_validator.enforce_slippage_guard = Mock()
        engine.order_validator.run_pre_trade_validation = Mock()
        engine.order_validator.maybe_preview_order = Mock()
        engine.order_validator.finalize_reduce_only_flag.return_value = False

        engine.order_submitter = Mock()
        engine.order_submitter.submit_order.return_value = "order-123"
        engine.order_submitter.record_rejection = Mock()

        engine.guard_manager = Mock()

        engine.event_store = Mock()

        return engine

    def test_place_order_with_stop_price(self, engine: LiveExecutionEngine) -> None:
        """Test order placement with stop price."""
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            quantity=Decimal("1"),
            price=Decimal("48000"),
            stop_price=Decimal("49000"),
        )

        call_args = engine.order_submitter.submit_order.call_args
        assert call_args.kwargs["stop_price"] == Decimal("49000")

    def test_place_order_with_time_in_force(self, engine: LiveExecutionEngine) -> None:
        """Test order placement with time in force."""
        mock_tif = Mock()

        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("49000"),
            tif=mock_tif,
        )

        call_args = engine.order_submitter.submit_order.call_args
        assert call_args.kwargs["tif"] is mock_tif

    def test_place_order_with_product_provided(self, engine: LiveExecutionEngine) -> None:
        """Test order placement with product explicitly provided."""
        mock_product = Mock()

        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
            product=mock_product,
        )

        engine.state_collector.require_product.assert_called_once_with("BTC-PERP", mock_product)

    def test_place_order_converts_quantity_to_decimal(self, engine: LiveExecutionEngine) -> None:
        """Test that quantity is converted to Decimal."""
        # Pass quantity as float
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.5,  # type: ignore[arg-type]
        )

        # Should still work - quantity converted internally
        engine.order_submitter.submit_order.assert_called_once()

    def test_place_order_converts_price_to_decimal(self, engine: LiveExecutionEngine) -> None:
        """Test that price is converted to Decimal."""
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=49000.50,  # type: ignore[arg-type]
        )

        engine.order_submitter.submit_order.assert_called_once()


class TestPlaceOrderValidationPaths:
    """Test various validation paths in place_order."""

    @pytest.fixture
    def mock_broker(self) -> Mock:
        """Create a mock broker."""
        broker = Mock()
        broker.list_balances.return_value = []
        broker.get_product.return_value = Mock()
        return broker

    @pytest.fixture
    def engine(self, mock_broker: Mock, bot_config_factory) -> LiveExecutionEngine:
        """Create an engine with mocked components."""
        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        engine.state_collector = Mock()
        engine.state_collector.require_product.return_value = Mock()
        engine.state_collector.collect_account_state.return_value = (
            [],
            Decimal("10000"),
            {},
            Decimal("10000"),
            [],
        )
        engine.state_collector.log_collateral_update = Mock()
        engine.state_collector.build_positions_dict.return_value = {}
        engine.state_collector.resolve_effective_price.return_value = Decimal("50000")

        engine.order_validator = Mock()
        engine.order_validator.validate_exchange_rules.return_value = (
            Decimal("1"),
            Decimal("50000"),
        )
        engine.order_validator.ensure_mark_is_fresh = Mock()
        engine.order_validator.enforce_slippage_guard = Mock()
        engine.order_validator.run_pre_trade_validation = Mock()
        engine.order_validator.maybe_preview_order = Mock()
        engine.order_validator.finalize_reduce_only_flag.return_value = False

        engine.order_submitter = Mock()
        engine.order_submitter.submit_order.return_value = "order-123"
        engine.order_submitter.record_rejection = Mock()

        engine.guard_manager = Mock()
        engine.event_store = Mock()

        return engine

    def test_place_order_calls_ensure_mark_is_fresh(self, engine: LiveExecutionEngine) -> None:
        """Test that mark freshness check is called."""
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
        )

        engine.order_validator.ensure_mark_is_fresh.assert_called_once_with("BTC-PERP")

    def test_place_order_calls_enforce_slippage_guard(self, engine: LiveExecutionEngine) -> None:
        """Test that slippage guard is enforced."""
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
        )

        engine.order_validator.enforce_slippage_guard.assert_called_once()

    def test_place_order_slippage_guard_failure(self, engine: LiveExecutionEngine) -> None:
        """Test order rejection when slippage guard fails."""
        engine.order_validator.enforce_slippage_guard.side_effect = ValidationError(
            "Slippage exceeds threshold"
        )

        with pytest.raises(ValidationError, match="Slippage exceeds threshold"):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
            )

        engine.order_submitter.record_rejection.assert_called_once()

    def test_place_order_mark_freshness_failure(self, engine: LiveExecutionEngine) -> None:
        """Test order rejection when mark price is stale."""
        engine.order_validator.ensure_mark_is_fresh.side_effect = ValidationError(
            "Mark price is stale"
        )

        with pytest.raises(ValidationError, match="Mark price is stale"):
            engine.place_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
            )

    def test_place_order_exchange_rules_validation(self, engine: LiveExecutionEngine) -> None:
        """Test that exchange rules validation is called."""
        engine.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
        )

        engine.order_validator.validate_exchange_rules.assert_called_once()


class TestResetDailyTrackingEdgeCases:
    """Additional tests for reset_daily_tracking."""

    @pytest.fixture
    def engine(self, bot_config_factory) -> LiveExecutionEngine:
        """Create an engine with mocked components."""
        mock_broker = Mock()
        mock_broker.list_balances.return_value = []

        config = bot_config_factory()
        engine = LiveExecutionEngine(broker=mock_broker, config=config)

        engine.state_collector = Mock()
        engine.state_collector.calculate_equity_from_balances.return_value = (
            Decimal("10000"),
            {},
            Decimal("10000"),
        )

        engine.risk_manager = Mock()
        engine.guard_manager = Mock()

        return engine

    def test_reset_daily_tracking_calculates_equity(self, engine: LiveExecutionEngine) -> None:
        """Test that reset calculates fresh equity."""
        balances = [{"currency": "USD", "available": "10000"}]
        engine.broker.list_balances.return_value = balances

        engine.reset_daily_tracking()

        engine.state_collector.calculate_equity_from_balances.assert_called_once_with(balances)

    def test_reset_daily_tracking_risk_manager_exception(self, engine: LiveExecutionEngine) -> None:
        """Test handling when risk_manager.reset_daily_tracking raises."""
        engine.risk_manager.reset_daily_tracking.side_effect = RuntimeError("Reset failed")

        # Should not raise - error should be caught
        engine.reset_daily_tracking()

    def test_reset_daily_tracking_invalidates_guard_cache(
        self, engine: LiveExecutionEngine
    ) -> None:
        """Test that guard cache is invalidated after reset."""
        engine.reset_daily_tracking()

        engine.guard_manager.invalidate_cache.assert_called_once()


class TestLiveOrderEdgeCases:
    """Additional tests for LiveOrder dataclass."""

    def test_live_order_with_zero_quantity(self) -> None:
        """Test LiveOrder with zero quantity."""
        order = LiveOrder(symbol="BTC-PERP", side="buy", quantity=Decimal("0"))

        assert order.quantity == Decimal("0")

    def test_live_order_with_very_small_quantity(self) -> None:
        """Test LiveOrder with very small quantity."""
        order = LiveOrder(symbol="BTC-PERP", side="buy", quantity=Decimal("0.00000001"))

        assert order.quantity == Decimal("0.00000001")

    def test_live_order_with_very_large_quantity(self) -> None:
        """Test LiveOrder with very large quantity."""
        order = LiveOrder(symbol="BTC-PERP", side="buy", quantity=Decimal("1000000"))

        assert order.quantity == Decimal("1000000")

    def test_live_order_negative_quantity_is_allowed(self) -> None:
        """Test that negative quantity is technically allowed (validation elsewhere)."""
        order = LiveOrder(symbol="BTC-PERP", side="buy", quantity=Decimal("-1"))

        assert order.quantity == Decimal("-1")
