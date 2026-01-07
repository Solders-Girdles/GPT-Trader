"""Tests for chaos testing harness."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.support.chaos import (
    ChaosBroker,
    FaultAction,
    FaultPlan,
    api_outage_scenario,
    broker_read_failures_scenario,
    fault_after,
    fault_always,
    fault_once,
    fault_sequence,
    preview_failures_scenario,
    slippage_spike_scenario,
)


class TestFaultAction:
    """Test FaultAction behavior."""

    def test_should_trigger_immediately(self) -> None:
        action = FaultAction(after_calls=0, times=1)
        assert action.should_trigger(1, "test") is True

    def test_should_not_trigger_before_after_calls(self) -> None:
        action = FaultAction(after_calls=3, times=1)
        assert action.should_trigger(1, "test") is False
        assert action.should_trigger(2, "test") is False
        assert action.should_trigger(3, "test") is True

    def test_should_not_trigger_after_times_exhausted(self) -> None:
        action = FaultAction(after_calls=0, times=2)
        assert action.should_trigger(1, "test") is True
        action._triggered = 1
        assert action.should_trigger(2, "test") is True
        action._triggered = 2
        assert action.should_trigger(3, "test") is False

    def test_should_trigger_forever_with_negative_times(self) -> None:
        action = FaultAction(after_calls=0, times=-1)
        action._triggered = 100
        assert action.should_trigger(101, "test") is True

    def test_execute_raises_exception(self) -> None:
        action = FaultAction(raise_exc=ValueError("test error"))
        with pytest.raises(ValueError, match="test error"):
            action.execute()

    def test_execute_returns_value(self) -> None:
        action = FaultAction(return_value={"data": "test"})
        assert action.execute() == {"data": "test"}

    def test_execute_with_delay(self) -> None:
        delays: list[float] = []
        action = FaultAction(return_value="ok", delay_seconds=0.5)
        action.execute(sleep_func=delays.append)
        assert delays == [0.5]

    def test_predicate_filters_calls(self) -> None:
        action = FaultAction(
            after_calls=0,
            times=1,
            predicate=lambda m, *a, **kw: kw.get("symbol") == "BTC-USD",
        )
        assert action.should_trigger(1, "get_ticker", symbol="BTC-USD") is True
        assert action.should_trigger(1, "get_ticker", symbol="ETH-USD") is False


class TestFaultPlan:
    """Test FaultPlan behavior."""

    def test_add_returns_self_for_chaining(self) -> None:
        plan = FaultPlan()
        result = plan.add("method1", FaultAction()).add("method2", FaultAction())
        assert result is plan
        assert "method1" in plan.faults and "method2" in plan.faults

    def test_apply_increments_call_count(self) -> None:
        plan = FaultPlan()
        plan.apply("test_method")
        plan.apply("test_method")
        assert plan.get_call_count("test_method") == 2

    def test_apply_returns_matching_action(self) -> None:
        plan = FaultPlan().add("test", FaultAction(raise_exc=ValueError()))
        should_fault, action = plan.apply("test")
        assert should_fault is True and action is not None

    def test_apply_returns_none_for_unmatched(self) -> None:
        plan = FaultPlan().add("other", FaultAction())
        should_fault, action = plan.apply("test")
        assert should_fault is False and action is None

    def test_reset_clears_state(self) -> None:
        plan = FaultPlan().add("test", FaultAction(times=1))
        plan.apply("test")
        plan.reset()
        assert plan.get_call_count("test") == 0


class TestChaosBroker:
    """Test ChaosBroker wrapping behavior."""

    def test_delegates_to_wrapped_when_no_fault(self) -> None:
        wrapped = MagicMock()
        wrapped.get_ticker.return_value = {"price": "50000"}
        broker = ChaosBroker(wrapped, FaultPlan())
        assert broker.get_ticker("BTC-USD") == {"price": "50000"}
        wrapped.get_ticker.assert_called_once_with("BTC-USD")

    def test_injects_fault_instead_of_delegating(self) -> None:
        wrapped = MagicMock()
        plan = FaultPlan().add("get_ticker", fault_once(raise_exc=TimeoutError("fail")))
        broker = ChaosBroker(wrapped, plan)
        with pytest.raises(TimeoutError):
            broker.get_ticker("BTC-USD")
        wrapped.get_ticker.assert_not_called()

    def test_returns_fault_value(self) -> None:
        wrapped = MagicMock()
        wrapped.get_ticker.return_value = {"price": "50000"}
        plan = FaultPlan().add("get_ticker", fault_once(return_value={"price": "0"}))
        broker = ChaosBroker(wrapped, plan)
        result = broker.get_ticker("BTC-USD")
        assert result == {"price": "0"}

    def test_fault_exhausted_delegates(self) -> None:
        wrapped = MagicMock()
        wrapped.get_ticker.return_value = {"price": "50000"}
        plan = FaultPlan().add("get_ticker", fault_once(return_value={"price": "0"}))
        broker = ChaosBroker(wrapped, plan)
        broker.get_ticker("BTC-USD")  # Fault triggers
        result = broker.get_ticker("BTC-USD")  # Delegates
        assert result == {"price": "50000"}

    def test_non_callable_attributes_passthrough(self) -> None:
        wrapped = MagicMock()
        wrapped.name = "TestBroker"
        broker = ChaosBroker(wrapped, FaultPlan())
        assert broker.name == "TestBroker"


class TestDeterministicHelpers:
    """Test helper functions for creating fault actions."""

    def test_fault_once_triggers_once(self) -> None:
        action = fault_once(raise_exc=ValueError())
        assert action.times == 1 and action.after_calls == 0

    def test_fault_after_delays_trigger(self) -> None:
        action = fault_after(5, raise_exc=ValueError())
        assert action.after_calls == 5

    def test_fault_always_triggers_forever(self) -> None:
        action = fault_always(return_value="test")
        assert action.times == -1

    def test_fault_sequence_creates_ordered_actions(self) -> None:
        actions = fault_sequence(
            [
                FaultAction(raise_exc=ValueError("first")),
                FaultAction(raise_exc=ValueError("second")),
            ]
        )
        assert len(actions) == 2
        assert actions[0].after_calls == 0
        assert actions[1].after_calls == 1


class TestScenarioPresets:
    """Test scenario preset factories."""

    def test_api_outage_scenario(self) -> None:
        plan = api_outage_scenario(error_rate=0.3, open_breakers=["orders", "quotes"])
        wrapped = MagicMock()
        broker = ChaosBroker(wrapped, plan)
        status = broker.get_resilience_status()
        assert status["metrics"]["error_rate"] == 0.3
        assert "orders" in status["circuit_breakers"]

    def test_slippage_spike_scenario(self) -> None:
        plan = slippage_spike_scenario(expected_bps=200)
        wrapped = MagicMock()
        broker = ChaosBroker(wrapped, plan)
        snapshot = broker.get_market_snapshot("BTC-USD")
        assert snapshot["spread_bps"] == 100  # Half of expected

    def test_preview_failures_scenario(self) -> None:
        plan = preview_failures_scenario(times=3)
        wrapped = MagicMock()
        broker = ChaosBroker(wrapped, plan)
        for _ in range(3):
            with pytest.raises(Exception, match="Preview service unavailable"):
                broker.preview_order(symbol="BTC-USD")
        # Fourth call should delegate
        wrapped.preview_order.return_value = {"preview": "ok"}
        assert broker.preview_order(symbol="BTC-USD") == {"preview": "ok"}

    def test_broker_read_failures_scenario(self) -> None:
        plan = broker_read_failures_scenario(times=2)
        wrapped = MagicMock()
        broker = ChaosBroker(wrapped, plan)
        for _ in range(2):
            with pytest.raises(ConnectionError):
                broker.list_balances()
        wrapped.list_balances.return_value = []
        assert broker.list_balances() == []
