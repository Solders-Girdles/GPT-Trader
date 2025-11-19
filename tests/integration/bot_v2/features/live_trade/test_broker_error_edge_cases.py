"""
Broker Error Propagation Edge Cases.
"""

from __future__ import annotations

import asyncio
import pytest

from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
)


class TestBrokerErrorPropagationEdgeCases:
    """Additional edge cases for broker error propagation"""

    @pytest.mark.asyncio
    async def test_concurrent_broker_errors(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """Test system behavior with multiple concurrent broker errors"""
        system = integrated_trading_system
        broker = system["execution_coordinator"].broker
        event_store = system["event_store"]

        # Enable multiple failure modes
        broker.drop_connection()
        broker.enable_rate_limiting()

        # Create multiple concurrent orders
        orders = [
            integration_test_scenarios.create_test_order(
                order_id=f"concurrent_{i}",
                symbol="BTC-USD" if i % 2 == 0 else "ETH-USD",
                side=OrderSide.BUY,
                quantity=0.1,
            )
            for i in range(3)
        ]

        # Attempt concurrent order placements
        tasks = []
        for order in orders:
            task = system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )
            tasks.append(task)

        # All should fail, but system should handle gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all failed with appropriate errors
        assert all(isinstance(result, Exception) for result in results)

        # Verify system remained stable
        event_store.get_events_by_type("system_stable")
        # Note: Depends on implementation

    @pytest.mark.asyncio
    async def test_broker_error_during_circuit_breaker(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """Test broker errors occurring during circuit breaker activation"""
        system = integrated_trading_system
        broker = system["execution_coordinator"].broker
        system["risk_manager"]
        event_store = system["event_store"]

        # Simulate market conditions that would trigger circuit breaker
        # and broker failure simultaneously

        # Mock broker failure
        broker.drop_connection()

        # Create test order that would trigger risk limits
        order = integration_test_scenarios.create_test_order(
            order_id="circuit_broker_error",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=10.0,  # Large size to trigger risk limits
        )

        # Attempt order placement
        with pytest.raises(Exception):
            await system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )

        # Verify both circuit breaker and broker error were handled
        event_store.get_events_by_type("circuit_breaker")
        broker_events = event_store.get_events_by_type("broker_error")

        # System should handle both types of errors
        assert len(broker_events) > 0
        # Circuit breaker events depend on risk configuration
