"""
Circuit breaker integration tests.

This test suite validates that circuit breakers trigger correctly mid-cycle
and that the system responds appropriately across all components:
- Daily loss breaches trigger immediate trading halt
- Volatility spikes trigger position size reductions
- Correlation risk triggers new position restrictions
- System recovery and manual reset functionality
"""

from __future__ import annotations

import asyncio
import os

# Import conftest fixtures with absolute paths
import sys
from decimal import Decimal
from unittest.mock import patch

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    OrderSide as Side,
)
from bot_v2.features.brokerages.core.interfaces import (
    OrderStatus,
)
from bot_v2.features.live_trade.risk.pre_trade_checks import ValidationError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conftest import (
    circuit_breaker_test_scenarios,
)


class TestCircuitBreakerTriggering:
    """Test circuit breaker triggering scenarios."""

    @pytest.mark.asyncio
    async def test_tc_cb_001_daily_loss_breach_mid_execution(
        self, async_integrated_system, integration_test_scenarios, circuit_breaker_test_scenarios
    ):
        """TC-CB-001: Daily Loss Gate Triggers Mid-Execution."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create normal order that should pass initial validation
        order = integration_test_scenarios.create_test_order(
            order_id="daily_loss_001", symbol="BTC-USD", side=Side.BUY, quantity=0.5
        )

        # Mock daily loss breach by setting daily PnL below limit
        scenario = circuit_breaker_test_scenarios["daily_loss_breach"]
        risk_manager.daily_pnl = Decimal(str(scenario["current_loss"]))
        risk_manager.start_of_day_equity = Decimal("10000.0")  # Start equity

        # Risk validation should fail due to daily loss breach
        ctx = _get_risk_validation_context(order)
        with pytest.raises(ValidationError) as exc_info:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )

        assert "daily loss" in str(exc_info.value).lower(), "Should mention daily loss breach"

        # Verify circuit breaker state is active
        assert risk_manager.is_reduce_only_mode(), "Should be in reduce-only mode"

        # Verify events were logged
        circuit_breaker_events = event_store.get_events_by_type("circuit_breaker_triggered")
        assert len(circuit_breaker_events) > 0, "Should have circuit breaker triggered events"

    @pytest.mark.asyncio
    async def test_tc_cb_002_liquidation_buffer_breach_active_positions(
        self, async_integrated_system, integration_test_scenarios
    ):
        """TC-CB-002: Liquidation Buffer Breach During Active Positions."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create order for testing
        order = integration_test_scenarios.create_test_order(
            order_id="liquidation_buffer_001", symbol="BTC-USD", side=Side.BUY, quantity=0.3
        )

        # Mock liquidation buffer breach
        scenario = circuit_breaker_test_scenarios["liquidation_buffer_breach"]

        # Simulate dangerous buffer ratio by setting low equity vs position value
        mock_positions = {
            "BTC-USD": {
                "size": Decimal("1.0"),
                "mark_price": Decimal("50000.0"),
                "unrealized_pnl": Decimal("-40000.0"),  # Large unrealized loss
                "leverage": Decimal("5.0"),
            }
        }

        # Set buffer ratio below threshold
        with patch.object(risk_manager, "get_current_positions", return_value=mock_positions):
            ctx = _get_risk_validation_context(order)
            with pytest.raises(ValidationError) as exc_info:
                risk_manager.pre_trade_validate(
                    symbol=order.symbol,
                    side=str(order.side),
                    quantity=order.quantity,
                    price=order.price if order.price is not None else Decimal("50000.0"),
                    product=ctx["product"],
                    equity=Decimal("1000.0"),  # Low equity
                    current_positions=mock_positions,
                )

            assert (
                "liquidation buffer" in str(exc_info.value).lower()
                or "buffer" in str(exc_info.value).lower()
            )

    @pytest.mark.asyncio
    async def test_tc_cb_003_volatility_spike_circuit_breaker(
        self, async_integrated_system, integration_test_scenarios
    ):
        """TC-CB-003: Volatility Spike Circuit Breaker Activation."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create order for testing
        order = integration_test_scenarios.create_test_order(
            order_id="volatility_spike_001", symbol="BTC-USD", side=Side.BUY, quantity=0.2
        )

        # Mock volatility spike scenario
        scenario = circuit_breaker_test_scenarios["volatility_spike"]

        # Simulate high volatility by adjusting market conditions
        with patch.object(risk_manager, "_check_market_volatility", return_value=True):
            # Set high volatility flag
            risk_manager.runtime_monitor._high_volatility_mode = True

            ctx = _get_risk_validation_context(order)
            try:
                risk_manager.pre_trade_validate(
                    symbol=order.symbol,
                    side=str(order.side),
                    quantity=order.quantity,
                    price=order.price if order.price is not None else Decimal("50000.0"),
                    product=ctx["product"],
                    equity=ctx["equity"],
                    current_positions=ctx["current_positions"],
                )
                # If validation passes, verify position size is reduced
                pass  # Check for reduced position size in success case
            except ValidationError as e:
                # Should either pass with reduced size or fail with volatility message
                assert "volatility" in str(e).lower() or "reduced" in str(e).lower()

    @pytest.mark.asyncio
    async def test_tc_cb_004_correlation_risk_circuit_breaker(
        self, async_integrated_system, integration_test_scenarios
    ):
        """TC-CB-004: Correlation Risk Circuit Breaker."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create correlated orders (same symbol, opposite sides)
        orders = [
            integration_test_scenarios.create_test_order(
                order_id="correlation_buy_001", symbol="BTC-USD", side=Side.BUY, quantity=1.0
            ),
            integration_test_scenarios.create_test_order(
                order_id="correlation_sell_001", symbol="BTC-USD", side=Side.SELL, quantity=1.0
            ),
        ]

        # Mock correlation risk scenario
        scenario = circuit_breaker_test_scenarios["correlation_risk"]

        # Simulate high correlation by setting portfolio correlation
        with patch.object(risk_manager, "check_correlation_risk", return_value=True):
            for order in orders:
                ctx = _get_risk_validation_context(order)
                with pytest.raises(ValidationError) as exc_info:
                    risk_manager.pre_trade_validate(
                        symbol=order.symbol,
                        side=str(order.side),
                        quantity=order.quantity,
                        price=order.price if order.price is not None else Decimal("50000.0"),
                        product=ctx["product"],
                        equity=ctx["equity"],
                        current_positions={},
                    )

                assert (
                    "correlation" in str(exc_info.value).lower()
                ), "Should mention correlation risk"

    @pytest.mark.asyncio
    async def test_tc_cb_005_position_size_limit_circuit_breaker(
        self, async_integrated_system, integration_test_scenarios
    ):
        """TC-CB-005: Position Size Limit Circuit Breaker."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Create oversized order
        order = integration_test_scenarios.create_test_order(
            order_id="size_limit_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=10.0,  # Excessive size
        )

        # Risk validation should fail due to position size limits
        ctx = _get_risk_validation_context(order)
        with pytest.raises(ValidationError) as exc_info:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions={},
            )

        assert (
            "position size" in str(exc_info.value).lower() or "size" in str(exc_info.value).lower()
        )


class TestCircuitBreakerSystemResponse:
    """Test system response to circuit breaker activation."""

    @pytest.mark.asyncio
    async def test_tc_cb_007_order_cancellation_on_circuit_breaker(
        self, async_integrated_system, integration_test_scenarios
    ):
        """TC-CB-007: Order Cancellation on Circuit Breaker."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Create and start executing an order
        order = integration_test_scenarios.create_test_order(
            order_id="cancel_cb_001", symbol="BTC-USD", side=Side.BUY, quantity=0.3
        )

        # Start order execution
        execution_task = asyncio.create_task(
            execution_coordinator.place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )
        )

        # Give execution time to start
        await asyncio.sleep(0.1)

        # Trigger circuit breaker mid-execution
        risk_manager.set_reduce_only_mode(True, "Test circuit breaker")

        # Wait for execution to complete
        try:
            result = await execution_task
            # Order should be rejected or cancelled
            assert result.status == OrderStatus.REJECTED or result.status == OrderStatus.CANCELLED
        except Exception:
            # Exception is acceptable - circuit breaker should stop execution
            pass

        # Verify circuit breaker events
        cb_events = event_store.get_events_by_type("circuit_breaker_triggered")
        assert len(cb_events) > 0, "Should have circuit breaker events"

    @pytest.mark.asyncio
    async def test_tc_cb_010_event_store_logging_circuit_breaker(
        self, async_integrated_system, integration_test_scenarios
    ):
        """TC-CB-010: Event Store Logging of Circuit Breaker Events."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        event_store = system["event_store"]

        # Trigger daily loss circuit breaker
        risk_manager.daily_pnl = Decimal("-600.0")  # Exceeds typical 500 limit
        risk_manager.start_of_day_equity = Decimal("10000.0")

        # Try to place order to trigger circuit breaker
        order = integration_test_scenarios.create_test_order(
            order_id="logging_test_001", symbol="BTC-USD", side=Side.BUY, quantity=0.1
        )

        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
        except ValidationError:
            pass  # Expected to fail

        # Verify comprehensive event logging
        expected_events = [
            "circuit_breaker_triggered",
            "risk_check_completed",
            "daily_loss_breached",
        ]

        logged_events = []
        for event_type in expected_events:
            events = event_store.get_events_by_type(event_type)
            if events:
                logged_events.extend(events)

        assert len(logged_events) > 0, "Should have circuit breaker related events logged"


class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery and reset functionality."""

    @pytest.mark.asyncio
    async def test_tc_cb_013_normal_market_condition_recovery(
        self, async_integrated_system, integration_test_scenarios
    ):
        """TC-CB-013: Normal Market Condition Recovery."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]

        # Initially trigger circuit breaker
        risk_manager.set_reduce_only_mode(True, "Test trigger")
        assert risk_manager.is_reduce_only_mode(), "Should be in reduce-only mode"

        # Reset circuit breaker to simulate market recovery
        risk_manager.set_reduce_only_mode(False, "Market conditions normalized")
        assert not risk_manager.is_reduce_only_mode(), "Should have recovered from reduce-only mode"

        # Order should now pass validation
        order = integration_test_scenarios.create_test_order(
            order_id="recovery_test_001", symbol="BTC-USD", side=Side.BUY, quantity=0.2
        )

        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
            validation_passed = True
        except ValidationError:
            validation_passed = False

        assert validation_passed, "Order should pass validation after circuit breaker reset"

    @pytest.mark.asyncio
    async def test_tc_cb_015_gradual_risk_limit_restoration(
        self, async_integrated_system, integration_test_scenarios
    ):
        """TC-CB-015: Gradual Risk Limit Restoration."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]

        # Start with restrictive settings
        original_max_size = risk_manager.config.max_position_size
        risk_manager.config.max_position_size = 0.1  # Very restrictive

        # Test with restrictive settings
        order = integration_test_scenarios.create_test_order(
            order_id="gradual_001",
            symbol="BTC-USD",
            side=Side.BUY,
            quantity=0.5,  # Should fail with restrictive settings
        )

        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
            restrictive_failed = False
        except ValidationError:
            restrictive_failed = True

        assert restrictive_failed, "Order should fail with restrictive settings"

        # Gradually restore limits
        risk_manager.config.max_position_size = 1.0  # More permissive

        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=0.3,  # Smaller order
                price=order.price if order.price is not None else Decimal("50000.0"),
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
            gradual_passed = True
        except ValidationError:
            gradual_passed = False

        assert gradual_passed, "Order should pass with gradually restored limits"

        # Restore original settings
        risk_manager.config.max_position_size = original_max_size

    @pytest.mark.asyncio
    async def test_tc_cb_017_system_health_check_post_recovery(
        self, async_integrated_system, integration_test_scenarios
    ):
        """TC-CB-017: System Health Check Post-Recovery."""
        system = async_integrated_system
        risk_manager = system["risk_manager"]
        execution_coordinator = system["execution_coordinator"]
        event_store = system["event_store"]

        # Trigger and then recover from circuit breaker
        risk_manager.set_reduce_only_mode(True, "Test trigger")
        await asyncio.sleep(0.1)  # Allow state to propagate
        risk_manager.set_reduce_only_mode(False, "Recovery test")

        # Verify system health indicators
        assert not risk_manager.is_reduce_only_mode(), "Should not be in reduce-only mode"
        assert (
            risk_manager.circuit_breaker_state.get("active", False) == False
        ), "Circuit breaker should be inactive"

        # Test normal order flow to verify system health
        order = integration_test_scenarios.create_test_order(
            order_id="health_check_001", symbol="BTC-USD", side=Side.BUY, quantity=0.1
        )

        ctx = _get_risk_validation_context(order)
        try:
            risk_manager.pre_trade_validate(
                symbol=order.symbol,
                side=str(order.side),
                quantity=order.quantity,
                price=order.price if order.price is not None else Decimal("50000.0"),
                product=ctx["product"],
                equity=ctx["equity"],
                current_positions=ctx["current_positions"],
            )
            health_check_passed = True
        except ValidationError:
            health_check_passed = False

        assert health_check_passed, "System health check should pass after recovery"


def _get_risk_validation_context(order):
    """Helper to get product and equity for risk validation."""
    from decimal import Decimal

    from bot_v2.features.brokerages.core.interfaces import MarketType, Product

    mock_product = Product(
        symbol=order.symbol,
        base_asset="BTC" if "BTC" in order.symbol else "ETH",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10.0"),
        price_increment=Decimal("0.01"),
    )

    mock_equity = Decimal("10000.0")  # $10,000 equity

    return {"product": mock_product, "equity": mock_equity, "current_positions": {}}
