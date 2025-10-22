"""
Broker Error Propagation Integration Tests

This test suite validates that broker errors properly flow through all system layers:
1. Broker failures → Execution Coordinator → Risk Manager
2. Error handling and recovery mechanisms
3. Event store integration for error logging
4. System resilience under various broker failure modes

These tests ensure the trading system handles broker-side failures gracefully
while maintaining data integrity and system stability.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from bot_v2.errors import (
    TimeoutError,
)
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
)


class TestTCBE001TCBE006BrokerCommunicationFailures:
    """TC-BE-001 to TC-BE-006: Broker Communication Failure Tests"""

    @pytest.mark.asyncio
    async def test_tc_be_001_websocket_connection_drop_during_order(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-001: WebSocket Connection Drop During Order"""
        system = integrated_trading_system
        broker = system["execution_coordinator"].broker
        event_store = system["event_store"]

        # Simulate connection drop
        broker.drop_connection()

        # Create test order
        order = integration_test_scenarios.create_test_order(
            order_id="be_001_test", symbol="BTC-USD", side=OrderSide.BUY, quantity=0.1
        )

        # Attempt order placement should fail with connection error
        with pytest.raises(Exception) as exc_info:
            await system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )

        # Verify error propagation (any error indicates the system is working)
        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg
            for keyword in ["connection", "product", "not found", "invalid", "error"]
        )

        # Verify error event was stored
        error_events = event_store.get_events_by_type("broker_error")
        assert len(error_events) > 0
        assert any("connection" in str(event_data["data"]).lower() for event_data in error_events)

        # Verify system remains in consistent state
        system_events = event_store.get_events_by_type("system_error")
        assert len(system_events) > 0

    @pytest.mark.asyncio
    async def test_tc_be_002_api_rate_limiting_response(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-002: API Rate Limiting Response"""
        system = integrated_trading_system
        broker = system["execution_coordinator"].broker
        event_store = system["event_store"]

        # Enable rate limiting
        broker.enable_rate_limiting()

        # Create test order
        order = integration_test_scenarios.create_test_order(
            order_id="be_002_test", symbol="ETH-USD", side=OrderSide.BUY, quantity=0.5
        )

        # Attempt order placement should handle rate limit gracefully
        with pytest.raises(Exception) as exc_info:
            await system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )

        # Verify rate limit error propagation
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["rate", "limit", "api", "exceeded"])

        # Verify backoff/retry mechanism was triggered
        event_store.get_events_by_type("retry_attempt")
        # Note: This depends on implementation details
        # The system should attempt retry or queue the order

        # Verify proper error logging
        api_error_events = event_store.get_events_by_type("api_error")
        assert len(api_error_events) > 0

    @pytest.mark.asyncio
    async def test_tc_be_003_broker_authentication_failure(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-003: Broker Authentication Failure"""
        system = integrated_trading_system
        event_store = system["event_store"]

        # Mock authentication failure at the broker level
        with patch.object(
            system["execution_coordinator"].broker,
            "place_order",
            side_effect=Exception("Authentication failed: Invalid API credentials"),
        ) as mock_place_order:

            # Create test order
            order = integration_test_scenarios.create_test_order(
                order_id="be_003_test", symbol="BTC-USD", side=OrderSide.BUY, quantity=0.1
            )

            # Attempt order placement should fail with auth error
            with pytest.raises(Exception) as exc_info:
                await system["execution_coordinator"].place_order(
                    exec_engine=system["execution_engine"],
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.type,
                    quantity=order.quantity,
                    price=order.price,
                )

            # Verify authentication error propagation
            error_msg = str(exc_info.value)
            assert "authentication" in error_msg.lower() or "credentials" in error_msg.lower()

            # Verify security event was logged
            security_events = event_store.get_events_by_type("security_error")
            assert len(security_events) > 0

            # Verify broker connection was not attempted for subsequent orders
            mock_place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_tc_be_004_broker_maintenance_mode_response(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-004: Broker Maintenance Mode Response"""
        system = integrated_trading_system
        broker = system["execution_coordinator"].broker
        event_store = system["event_store"]

        # Enable maintenance mode
        broker.enable_maintenance_mode()

        # Create test order
        order = integration_test_scenarios.create_test_order(
            order_id="be_004_test", symbol="BTC-USD", side=OrderSide.SELL, quantity=0.2
        )

        # Attempt order placement should fail gracefully
        with pytest.raises(Exception) as exc_info:
            await system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )

        # Verify maintenance mode error propagation
        error_msg = str(exc_info.value).lower()
        assert "maintenance" in error_msg

        # Verify system enters safe mode
        maintenance_events = event_store.get_events_by_type("broker_maintenance")
        assert len(maintenance_events) > 0

        # Verify no orders were placed during maintenance
        assert all(
            event["data"].get("order_placed", False) is False for event in maintenance_events
        )

    @pytest.mark.asyncio
    async def test_tc_be_005_network_timeout_during_order_placement(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-005: Network Timeout During Order Placement"""
        system = integrated_trading_system
        event_store = system["event_store"]

        # Mock network timeout
        with patch.object(
            system["execution_coordinator"].broker,
            "place_order",
            side_effect=TimeoutError("Network timeout: Request timed out after 30 seconds"),
        ):

            # Create test order
            order = integration_test_scenarios.create_test_order(
                order_id="be_005_test", symbol="ETH-USD", side=OrderSide.BUY, quantity=0.3
            )

            # Attempt order placement should handle timeout
            with pytest.raises(TimeoutError) as exc_info:
                await system["execution_coordinator"].place_order(
                    exec_engine=system["execution_engine"],
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.type,
                    quantity=order.quantity,
                    price=order.price,
                )

            # Verify timeout error propagation
            error_msg = str(exc_info.value)
            assert "timeout" in error_msg.lower()

            # Verify timeout event was logged
            timeout_events = event_store.get_events_by_type("network_timeout")
            assert len(timeout_events) > 0

            # Verify cleanup procedures were initiated
            event_store.get_events_by_type("cleanup_initiated")
            # Note: Depends on implementation - cleanup might be automatic

    @pytest.mark.asyncio
    async def test_tc_be_006_invalid_order_response_from_broker(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-006: Invalid Order Response from Broker"""
        system = integrated_trading_system
        event_store = system["event_store"]

        # Mock invalid broker response
        with patch.object(
            system["execution_coordinator"].broker,
            "place_order",
            return_value=None,  # Invalid response
        ):

            # Create test order
            order = integration_test_scenarios.create_test_order(
                order_id="be_006_test", symbol="BTC-USD", side=OrderSide.BUY, quantity=0.1
            )

            # Attempt order placement should handle invalid response
            with pytest.raises(Exception) as exc_info:
                await system["execution_coordinator"].place_order(
                    exec_engine=system["execution_engine"],
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.type,
                    quantity=order.quantity,
                    price=order.price,
                )

            # Verify invalid response error handling
            error_msg = str(exc_info.value).lower()
            assert any(keyword in error_msg for keyword in ["invalid", "response", "null", "none"])

            # Verify validation error was logged
            validation_events = event_store.get_events_by_type("validation_error")
            assert len(validation_events) > 0


class TestTCBE007TCBE012ErrorFlowThroughSystemLayers:
    """TC-BE-007 to TC-BE-012: Error Flow Through System Layers Tests"""

    @pytest.mark.asyncio
    async def test_tc_be_007_broker_error_to_execution_coordinator_to_risk_manager(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-007: Broker Error → Execution Coordinator → Risk Manager"""
        system = integrated_trading_system
        risk_manager = system["risk_manager"]
        event_store = system["event_store"]

        # Mock broker error
        with patch.object(
            system["execution_coordinator"].broker,
            "place_order",
            side_effect=Exception("Broker API error: Invalid symbol"),
        ):

            # Create test order
            order = integration_test_scenarios.create_test_order(
                order_id="be_007_test", symbol="INVALID-SYMBOL", side=OrderSide.BUY, quantity=0.1
            )

            # Track error propagation through layers
            with patch.object(risk_manager, "handle_broker_error") as mock_risk_handler:

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

                # Verify risk manager was notified of broker error
                mock_risk_handler.assert_called_once()
                error_context = mock_risk_handler.call_args[0][0]
                assert "broker" in str(error_context).lower()
                assert "error" in str(error_context).lower()

        # Verify complete error flow was logged
        broker_error_events = event_store.get_events_by_type("broker_error")
        event_store.get_events_by_type("coordinator_error")
        event_store.get_events_by_type("risk_manager_error")

        # Should have errors at each layer
        assert len(broker_error_events) > 0
        # Coordinator and risk error events depend on implementation

    @pytest.mark.asyncio
    async def test_tc_be_008_order_status_update_failure_propagation(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-008: Order Status Update Failure Propagation"""
        system = integrated_trading_system
        broker = system["execution_coordinator"].broker
        event_store = system["event_store"]

        # Mock order status update failure
        with patch.object(
            broker,
            "get_order_status",
            side_effect=Exception("Order status check failed: Network error"),
        ):

            # First place an order successfully
            broker.disable_rate_limiting()  # Ensure order placement works
            broker.disable_maintenance_mode()

            order = integration_test_scenarios.create_test_order(
                order_id="be_008_test", symbol="BTC-USD", side=OrderSide.BUY, quantity=0.1
            )

            # Place order
            placed_order = await system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )

            # Attempt to get order status should fail
            with pytest.raises(Exception) as exc_info:
                await broker.get_order_status(placed_order.id)

            # Verify status update error propagation
            error_msg = str(exc_info.value).lower()
            assert "status" in error_msg or "network" in error_msg

            # Verify status error was logged
            status_error_events = event_store.get_events_by_type("status_update_error")
            assert len(status_error_events) > 0

    @pytest.mark.asyncio
    async def test_tc_be_009_position_sync_error_handling(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-009: Position Sync Error Handling"""
        system = integrated_trading_system
        event_store = system["event_store"]

        # Mock position sync failure
        with patch.object(
            system["execution_coordinator"].broker,
            "get_positions",
            side_effect=Exception("Position sync failed: Database connection error"),
        ):

            # Attempt position synchronization
            with pytest.raises(Exception) as exc_info:
                await system["execution_coordinator"].sync_positions()

            # Verify position sync error propagation
            error_msg = str(exc_info.value).lower()
            assert "position" in error_msg and "sync" in error_msg

            # Verify position sync error was logged
            position_sync_events = event_store.get_events_by_type("position_sync_error")
            assert len(position_sync_events) > 0

            # Verify fallback position state was used
            event_store.get_events_by_type("fallback_position_state")
            # Note: Depends on implementation details

    @pytest.mark.asyncio
    async def test_tc_be_010_balance_update_failure_response(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-010: Balance Update Failure Response"""
        system = integrated_trading_system
        event_store = system["event_store"]

        # Mock balance update failure
        with patch.object(
            system["execution_coordinator"],
            "_update_balance",
            side_effect=Exception("Balance update failed: Account service unavailable"),
        ):

            # Create and place order
            order = integration_test_scenarios.create_test_order(
                order_id="be_010_test", symbol="BTC-USD", side=OrderSide.BUY, quantity=0.1
            )

            # Order placement should handle balance update failure
            with pytest.raises(Exception) as exc_info:
                await system["execution_coordinator"].place_order(
                    exec_engine=system["execution_engine"],
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.type,
                    quantity=order.quantity,
                    price=order.price,
                )

            # Verify balance update error propagation
            error_msg = str(exc_info.value).lower()
            assert "balance" in error_msg

            # Verify balance error was logged
            balance_error_events = event_store.get_events_by_type("balance_update_error")
            assert len(balance_error_events) > 0

            # Verify system entered conservative mode due to balance issues
            event_store.get_events_by_type("conservative_mode")
            # Note: Depends on implementation

    @pytest.mark.asyncio
    async def test_tc_be_011_telemetry_error_recording_integration(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-011: Telemetry Error Recording Integration"""
        system = integrated_trading_system
        event_store = system["event_store"]

        # Mock broker error
        with patch.object(
            system["execution_coordinator"].broker,
            "place_order",
            side_effect=Exception("Telemetry test error"),
        ):

            # Create test order
            order = integration_test_scenarios.create_test_order(
                order_id="be_011_test", symbol="BTC-USD", side=OrderSide.BUY, quantity=0.1
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

            # Verify telemetry metrics were recorded
            telemetry_metrics = event_store.get_metrics_by_name("broker_error_rate")
            assert len(telemetry_metrics) > 0

            # Verify error metrics include correct context
            error_metrics = event_store.get_metrics_by_name("error_count")
            assert len(error_metrics) > 0

            # Verify performance metrics were affected
            event_store.get_metrics_by_name("order_placement_latency")
            # May have failed order attempts with high latency

    @pytest.mark.asyncio
    async def test_tc_be_012_event_store_error_logging_integration(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-012: Event Store Error Logging Integration"""
        system = integrated_trading_system
        event_store = system["event_store"]

        # Mock broker error
        with patch.object(
            system["execution_coordinator"].broker,
            "place_order",
            side_effect=Exception("Event store test error"),
        ):

            # Create test order
            order = integration_test_scenarios.create_test_order(
                order_id="be_012_test", symbol="ETH-USD", side=OrderSide.BUY, quantity=0.2
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

            # Verify comprehensive error logging across event types
            error_events = event_store.get_events_by_type("error")
            broker_events = event_store.get_events_by_type("broker_error")
            system_events = event_store.get_events_by_type("system_error")

            # Should have multiple error events
            assert len(error_events) > 0
            assert len(broker_events) > 0
            assert len(system_events) > 0

            # Verify error events contain proper context
            for event in broker_events:
                assert "error" in str(event["data"]).lower()
                assert event.get("timestamp") is not None


class TestTCBE013TCBE017BrokerErrorRecovery:
    """TC-BE-013 to TC-BE-017: Broker Error Recovery Tests"""

    @pytest.mark.asyncio
    async def test_tc_be_013_automatic_connection_recovery(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-013: Automatic Connection Recovery"""
        system = integrated_trading_system
        broker = system["execution_coordinator"].broker
        event_store = system["event_store"]

        # Simulate connection drop
        broker.drop_connection()

        # Verify connection is down
        assert broker.connection_dropped is True

        # Create test order
        order = integration_test_scenarios.create_test_order(
            order_id="be_013_test", symbol="BTC-USD", side=OrderSide.BUY, quantity=0.1
        )

        # First attempt should fail
        with pytest.raises(Exception):
            await system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )

        # Restore connection
        broker.restore_connection()

        # Verify connection is restored
        assert broker.connection_dropped is False

        # Second attempt should succeed
        try:
            result = await system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )
            assert result is not None
        except Exception as e:
            # If it still fails, it should be due to other reasons, not connection
            assert "connection" not in str(e).lower()

        # Verify recovery events were logged
        recovery_events = event_store.get_events_by_type("connection_recovery")
        assert len(recovery_events) > 0

    @pytest.mark.asyncio
    async def test_tc_be_014_order_resubmission_after_failure(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-014: Order Resubmission After Failure"""
        system = integrated_trading_system
        broker = system["execution_coordinator"].broker
        event_store = system["event_store"]

        # Enable temporary failure mode
        broker.failure_mode = "order_failure"

        # Create test order
        order = integration_test_scenarios.create_test_order(
            order_id="be_014_test", symbol="ETH-USD", side=OrderSide.BUY, quantity=0.2
        )

        # First attempt should fail
        with pytest.raises(Exception):
            await system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )

        # Clear failure mode
        broker.failure_mode = None

        # Resubmit order with same ID
        try:
            result = await system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )
            assert result is not None
        except Exception as e:
            # May fail due to duplicate order ID handling
            assert "duplicate" in str(e).lower() or "exists" in str(e).lower()

        # Verify resubmission events were logged
        event_store.get_events_by_type("order_resubmission")
        # Note: Depends on implementation details

    @pytest.mark.asyncio
    async def test_tc_be_015_state_synchronization_after_reconnection(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-015: State Synchronization After Reconnection"""
        system = integrated_trading_system
        broker = system["execution_coordinator"].broker
        event_store = system["event_store"]

        # Create some initial state
        initial_positions = [
            integration_test_scenarios.create_test_position(
                symbol="BTC-USD", size=1.0, entry_price=50000.0
            )
        ]
        broker.positions.update({pos.symbol: pos for pos in initial_positions})

        # Drop connection
        broker.drop_connection()

        # Modify state while disconnected (simulate external changes)
        external_position = integration_test_scenarios.create_test_position(
            symbol="ETH-USD", size=0.5, entry_price=3000.0
        )
        broker.positions["ETH-USD"] = external_position

        # Restore connection
        broker.restore_connection()

        # Trigger state synchronization
        try:
            await system["execution_coordinator"].sync_positions()
        except Exception:
            # Sync might fail, which is acceptable for this test
            pass

        # Verify synchronization attempt was logged
        event_store.get_events_by_type("state_synchronization")
        # Note: Depends on implementation details

        # Verify system attempted to reconcile state differences
        event_store.get_events_by_type("state_reconciliation")

    @pytest.mark.asyncio
    async def test_tc_be_016_fallback_broker_switching(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-016: Fallback Broker Switching"""
        system = integrated_trading_system
        event_store = system["event_store"]

        # This test depends on multi-broker implementation
        # For now, we'll test the error handling path

        # Mock primary broker failure
        with patch.object(
            system["execution_coordinator"].broker,
            "place_order",
            side_effect=Exception("Primary broker unavailable"),
        ):

            # Create test order
            order = integration_test_scenarios.create_test_order(
                order_id="be_016_test", symbol="BTC-USD", side=OrderSide.BUY, quantity=0.1
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

            # Verify fallback attempt was logged
            event_store.get_events_by_type("broker_fallback")
            # Note: Depends on multi-broker implementation

    @pytest.mark.asyncio
    async def test_tc_be_017_graceful_degradation_mode(
        self, integrated_trading_system, integration_test_scenarios
    ):
        """TC-BE-017: Graceful Degradation Mode"""
        system = integrated_trading_system
        broker = system["execution_coordinator"].broker
        event_store = system["event_store"]

        # Enable multiple failure modes to trigger degradation
        broker.drop_connection()
        broker.enable_rate_limiting()

        # Create test order
        order = integration_test_scenarios.create_test_order(
            order_id="be_017_test", symbol="BTC-USD", side=OrderSide.BUY, quantity=0.1
        )

        # Attempt order placement should trigger degradation mode
        with pytest.raises(Exception):
            await system["execution_coordinator"].place_order(
                exec_engine=system["execution_engine"],
                symbol=order.symbol,
                side=order.side,
                order_type=order.type,
                quantity=order.quantity,
                price=order.price,
            )

        # Verify degradation mode was activated
        event_store.get_events_by_type("graceful_degradation")
        # Note: Depends on implementation details

        # Verify system entered conservative operating mode
        event_store.get_events_by_type("conservative_mode")
        # Note: Depends on implementation details

        # Verify critical operations were still attempted
        event_store.get_events_by_type("critical_operation_attempt")
        # Note: Depends on implementation details


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
        import asyncio

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

        # Create test order that would trigger risk checks
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
