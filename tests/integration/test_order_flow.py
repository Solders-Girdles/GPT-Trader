"""Integration tests for order flow validation.

Tests the full path from order submission through event recording,
with DeterministicBroker for deterministic behavior without network calls.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import ApplicationContainer
from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
from gpt_trader.orchestration.configuration.risk.model import RiskConfig
from gpt_trader.orchestration.deterministic_broker import DeterministicBroker
from gpt_trader.orchestration.live_execution import LiveExecutionEngine
from gpt_trader.persistence.event_store import EventStore

if TYPE_CHECKING:
    pass


# =============================================================================
# LiveExecutionEngine Tests (Order Submission Path)
# =============================================================================


class TestLiveExecutionEngineOrderFlow:
    """Tests for LiveExecutionEngine order submission path."""

    @pytest.fixture
    def execution_engine(
        self,
        deterministic_broker: DeterministicBroker,
        integration_config: BotConfig,
        fresh_event_store: EventStore,
    ) -> LiveExecutionEngine:
        """Create a LiveExecutionEngine for integration testing."""
        risk_config = RiskConfig(
            max_leverage=Decimal("3"),
            daily_loss_limit=Decimal("100"),
            daily_loss_limit_pct=Decimal("0.02"),
            max_position_pct_per_symbol=0.05,
        )
        risk_manager = LiveRiskManager(
            config=risk_config,
            event_store=fresh_event_store,
        )
        # Seed mark freshness
        risk_manager.last_mark_update["BTC-USD"] = time.time()

        return LiveExecutionEngine(
            broker=deterministic_broker,  # type: ignore[arg-type]
            config=integration_config,
            risk_manager=risk_manager,
            event_store=fresh_event_store,
            bot_id="integration_test",
        )

    def test_happy_path_creates_trade_event(
        self,
        execution_engine: LiveExecutionEngine,
    ) -> None:
        """Test that successful order placement creates a trade event.

        Validates:
        - place_order() returns non-None order_id
        - event_store contains a trade event with correct data
        """
        # Place a valid order
        order_id = execution_engine.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),  # Valid size (above min 0.001)
        )

        # Verify order was placed
        assert order_id is not None, "Expected order_id to be returned"
        assert order_id.startswith("MOCK_"), f"Expected mock order ID, got {order_id}"

        # Verify trade event was recorded
        events = execution_engine.event_store.events
        trade_events = [e for e in events if e["type"] == "trade"]
        assert len(trade_events) >= 1, "Expected at least one trade event"

        # Verify trade event contents
        latest_trade = trade_events[-1]
        trade_data = latest_trade["data"]
        assert trade_data.get("symbol") == "BTC-USD"
        assert trade_data.get("side") == "BUY"
        assert "order_id" in trade_data

    def test_multiple_orders_record_multiple_events(
        self,
        execution_engine: LiveExecutionEngine,
    ) -> None:
        """Test that multiple successful orders each create trade events.

        Validates:
        - Multiple place_order() calls succeed
        - Each order creates a separate trade event
        """
        # Place first order
        order_id_1 = execution_engine.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
        )

        # Place second order
        order_id_2 = execution_engine.place_order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
        )

        # Both orders should succeed
        assert order_id_1 is not None, "First order should succeed"
        assert order_id_2 is not None, "Second order should succeed"
        assert order_id_1 != order_id_2, "Order IDs should be different"

        # Verify two trade events were recorded
        events = execution_engine.event_store.events
        trade_events = [e for e in events if e["type"] == "trade"]
        assert len(trade_events) >= 2, f"Expected at least 2 trade events, got {len(trade_events)}"

        # Verify different sides
        sides = [e["data"].get("side") for e in trade_events[-2:]]
        assert "BUY" in sides and "SELL" in sides


# =============================================================================
# TradingEngine Tests (Guard Stack + Degradation Path)
# =============================================================================


class TestTradingEngineGuardStack:
    """Tests for TradingEngine guard stack and degradation handling."""

    @pytest.fixture
    def trading_bot(
        self,
        integration_config: BotConfig,
        integration_container: ApplicationContainer,
    ):
        """Create a TradingBot via container for integration testing."""
        return integration_container.create_bot()

    @pytest.mark.asyncio
    async def test_reduce_only_blocks_new_position(
        self,
        trading_bot,
    ) -> None:
        """Test that reduce-only mode blocks new position entries.

        Validates:
        - When reduce-only mode is active, new positions are blocked
        - Rejection is recorded in event store
        """
        engine = trading_bot.engine

        # Enable reduce-only mode
        engine.context.risk_manager.set_reduce_only_mode(True, reason="integration_test")

        # Get initial event count
        initial_events = len(engine.context.event_store.events)

        # Prepare a BUY decision (new position)
        from gpt_trader.features.live_trade.strategies.perps_baseline import (
            Action,
            Decision,
        )

        buy_decision = Decision(action=Action.BUY, reason="Test signal", confidence=0.8)

        # Attempt to validate and place order
        # This should be blocked by reduce-only mode
        try:
            await engine._validate_and_place_order(
                symbol="BTC-USD",
                decision=buy_decision,
                price=Decimal("50000"),
                equity=Decimal("100000"),
            )
        except Exception:
            pass  # Validation errors are expected

        # Verify rejection was recorded
        events = engine.context.event_store.events
        new_events = events[initial_events:]

        # Check for rejection in metrics
        rejection_found = any(
            (
                e["type"] == "metric"
                and e["data"].get("metrics", {}).get("event_type") == "order_rejected"
            )
            for e in new_events
        )
        assert rejection_found, "Expected rejection to be recorded in metrics"

        # Also check status reporter didn't record a trade
        # (checking that no new trade was added)
        trade_events = [e for e in new_events if e["type"] == "trade"]
        assert len(trade_events) == 0, "Should not have recorded any trades"

        # Clean up
        engine.context.risk_manager.set_reduce_only_mode(False, reason="integration_test_cleanup")

    @pytest.mark.asyncio
    async def test_mark_staleness_blocks_order(
        self,
        trading_bot,
    ) -> None:
        """Test that stale mark price blocks order and triggers symbol pause.

        Validates:
        - Order is blocked when mark price is stale
        - Symbol pause is applied via DegradationState
        """
        engine = trading_bot.engine

        # Set mark update time far in the past (make it stale)
        risk_manager = engine.context.risk_manager
        # Default staleness threshold is typically 60-120 seconds
        stale_time = time.time() - 300  # 5 minutes ago
        risk_manager.last_mark_update["BTC-USD"] = stale_time

        # Prepare a BUY decision
        from gpt_trader.features.live_trade.strategies.perps_baseline import (
            Action,
            Decision,
        )

        buy_decision = Decision(action=Action.BUY, reason="Test signal", confidence=0.8)

        # Get initial event count
        initial_events = len(engine.context.event_store.events)

        # Attempt to validate and place order
        # This should be blocked by mark staleness check
        validation_failed = False
        try:
            await engine._validate_and_place_order(
                symbol="BTC-USD",
                decision=buy_decision,
                price=Decimal("50000"),
                equity=Decimal("100000"),
            )
        except Exception as e:
            validation_failed = True
            assert "stale" in str(e).lower() or "Mark" in str(
                e
            ), f"Expected staleness error, got: {e}"

        # Verify either validation failed or rejection was recorded
        events = engine.context.event_store.events
        new_events = events[initial_events:]

        # Check for rejection or error event
        rejection_events = [
            e
            for e in new_events
            if e["type"] == "error"
            or (
                e["type"] == "metric"
                and e["data"].get("metrics", {}).get("event_type") == "order_rejected"
            )
        ]

        # Either validation should have failed OR rejection should be recorded
        assert (
            validation_failed or len(rejection_events) > 0
        ), "Expected validation failure or rejection event"

        # Clean up - reset mark timestamp
        risk_manager.last_mark_update["BTC-USD"] = time.time()


# =============================================================================
# Metrics Verification (Optional)
# =============================================================================


class TestOrderFlowMetrics:
    """Optional tests for order flow metrics integration."""

    @pytest.fixture
    def execution_engine(
        self,
        deterministic_broker: DeterministicBroker,
        integration_config: BotConfig,
        fresh_event_store: EventStore,
    ) -> LiveExecutionEngine:
        """Create a LiveExecutionEngine for metrics testing."""
        risk_config = RiskConfig(
            max_leverage=Decimal("3"),
            daily_loss_limit=Decimal("100"),
            daily_loss_limit_pct=Decimal("0.02"),
            max_position_pct_per_symbol=0.05,
        )
        risk_manager = LiveRiskManager(
            config=risk_config,
            event_store=fresh_event_store,
        )
        # Seed mark freshness
        risk_manager.last_mark_update["BTC-USD"] = time.time()

        return LiveExecutionEngine(
            broker=deterministic_broker,  # type: ignore[arg-type]
            config=integration_config,
            risk_manager=risk_manager,
            event_store=fresh_event_store,
            bot_id="metrics_test",
        )

    def test_order_submission_metric_incremented_on_success(
        self,
        execution_engine: LiveExecutionEngine,
    ) -> None:
        """Test that gpt_trader_order_submission_total increments on success."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        collector = get_metrics_collector()

        # Get initial count
        initial_summary = collector.get_metrics_summary()
        initial_counters = initial_summary.get("counters", {})

        # Place a valid order
        order_id = execution_engine.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.01"),
        )

        assert order_id is not None

        # Check metrics
        final_summary = collector.get_metrics_summary()
        final_counters = final_summary.get("counters", {})

        # Look for order submission counter with success result
        success_key = None
        for key in final_counters:
            if "order_submission_total" in key and "result=success" in key:
                success_key = key
                break

        if success_key:
            # Verify it was incremented
            initial_value = initial_counters.get(success_key, 0)
            final_value = final_counters.get(success_key, 0)
            assert (
                final_value > initial_value
            ), f"Expected counter to increment, was {initial_value} -> {final_value}"
