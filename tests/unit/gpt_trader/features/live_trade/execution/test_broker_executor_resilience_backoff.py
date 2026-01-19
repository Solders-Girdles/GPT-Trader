"""Deterministic resilience tests for BrokerExecutor (backoff behavior)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderType,
)
from gpt_trader.features.live_trade.execution.broker_executor import (
    BrokerExecutor,
    RetryPolicy,
)


class TestBrokerExecutorResilienceBackoff:
    """Deterministic resilience tests using failure injection harness."""

    def test_exponential_backoff_delays(self, sample_order: Order) -> None:
        """Test that retry delays follow exponential backoff."""
        from tests.fixtures.failure_injection import (
            FailureScript,
            InjectingBroker,
            counting_sleep,
        )

        mock_broker = MagicMock()
        mock_broker.place_order.return_value = sample_order
        script = FailureScript.fail_then_succeed(failures=3)
        injecting = InjectingBroker(mock_broker, place_order=script)

        sleep_fn, get_sleeps = counting_sleep()
        executor = BrokerExecutor(
            broker=injecting,
            retry_policy=RetryPolicy(
                max_attempts=5,
                base_delay=0.5,
                max_delay=10.0,
                jitter=0,  # No jitter for determinism
            ),
            sleep_fn=sleep_fn,
        )

        executor.execute_order(
            submit_id="backoff-test-1",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            use_retry=True,
        )

        sleeps = get_sleeps()
        assert len(sleeps) == 3
        # Exponential backoff: 0.5, 1.0, 2.0 (base * 2^(attempt-2))
        assert sleeps[0] == 0.5  # Attempt 2
        assert sleeps[1] == 1.0  # Attempt 3
        assert sleeps[2] == 2.0  # Attempt 4

    def test_non_retryable_mid_sequence_stops_immediately(self) -> None:
        """Test that non-retryable error mid-sequence stops retries immediately."""
        from tests.fixtures.failure_injection import (
            FailureScript,
            InjectingBroker,
            counting_sleep,
        )

        mock_broker = MagicMock()
        # Sequence: retryable fail, retryable fail, then non-retryable
        script = FailureScript.from_outcomes(
            ConnectionError("transient 1"),
            ConnectionError("transient 2"),
            ValueError("insufficient funds"),  # non-retryable
            None,  # would succeed, but should never reach here
        )
        injecting = InjectingBroker(mock_broker, place_order=script)

        sleep_fn, get_sleeps = counting_sleep()
        executor = BrokerExecutor(
            broker=injecting,
            retry_policy=RetryPolicy(max_attempts=5, base_delay=0.1, jitter=0),
            sleep_fn=sleep_fn,
        )

        with pytest.raises(ValueError, match="insufficient funds"):
            executor.execute_order(
                submit_id="mid-stop-test-1",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                stop_price=None,
                tif=None,
                reduce_only=False,
                leverage=None,
                use_retry=True,
            )

        # Should stop at attempt 3 (the ValueError), not continue to attempt 4
        assert injecting.get_call_count("place_order") == 3
        # Only 2 sleeps (before attempts 2 and 3)
        assert len(get_sleeps()) == 2
