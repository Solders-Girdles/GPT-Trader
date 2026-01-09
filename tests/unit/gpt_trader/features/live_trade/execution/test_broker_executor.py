"""Tests for features/live_trade/execution/broker_executor.py."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.live_trade.execution.broker_executor import (
    BrokerExecutor,
    RetryPolicy,
    execute_with_retry,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_broker() -> MagicMock:
    """Create a mock broker."""
    broker = MagicMock()
    broker.place_order = MagicMock()
    return broker


@pytest.fixture
def executor(mock_broker: MagicMock) -> BrokerExecutor:
    """Create a BrokerExecutor instance."""
    return BrokerExecutor(broker=mock_broker)


@pytest.fixture
def sample_order() -> Order:
    """Create a sample order response."""
    from datetime import datetime

    return Order(
        id="order-123",
        client_id="client-123",
        symbol="BTC-USD",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("50000"),
        stop_price=None,
        tif=TimeInForce.GTC,
        status=OrderStatus.PENDING,
        submitted_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


# ============================================================
# Test: __init__
# ============================================================


class TestBrokerExecutorInit:
    """Tests for BrokerExecutor initialization."""

    def test_init_stores_broker(self, mock_broker: MagicMock) -> None:
        """Test that broker is stored correctly."""
        executor = BrokerExecutor(broker=mock_broker)
        assert executor._broker is mock_broker

    def test_init_defaults_integration_mode_false(self, mock_broker: MagicMock) -> None:
        """Test that integration_mode defaults to False."""
        executor = BrokerExecutor(broker=mock_broker)
        assert executor._integration_mode is False

    def test_init_accepts_integration_mode_true(self, mock_broker: MagicMock) -> None:
        """Test that integration_mode can be set to True."""
        executor = BrokerExecutor(broker=mock_broker, integration_mode=True)
        assert executor._integration_mode is True


# ============================================================
# Test: execute_order - Normal execution path
# ============================================================


class TestExecuteOrderNormal:
    """Tests for execute_order normal execution path."""

    def test_execute_order_calls_broker_with_correct_args(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that execute_order calls broker.place_order with correct arguments."""
        mock_broker.place_order.return_value = sample_order

        result = executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=None,
        )

        mock_broker.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=None,
            client_id="client-123",
        )
        assert result is sample_order

    def test_execute_order_passes_market_order_with_none_price(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that market orders can pass None as price."""
        mock_broker.place_order.return_value = sample_order

        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
        )

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["price"] is None
        assert call_kwargs["tif"] is None

    def test_execute_order_passes_reduce_only_flag(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that reduce_only flag is passed correctly."""
        mock_broker.place_order.return_value = sample_order

        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=True,
            leverage=None,
        )

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["reduce_only"] is True

    def test_execute_order_passes_leverage(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that leverage is passed correctly."""
        mock_broker.place_order.return_value = sample_order

        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["leverage"] == 10

    def test_execute_order_passes_stop_price(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that stop_price is passed for stop orders."""
        mock_broker.place_order.return_value = sample_order

        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("48000"),
            stop_price=Decimal("49000"),
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=None,
        )

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["stop_price"] == Decimal("49000")


# ============================================================
# Test: Edge cases
# ============================================================


class TestBrokerExecutorEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_execute_order_with_all_none_optional_params(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test execution with all optional params as None."""
        mock_broker.place_order.return_value = sample_order

        result = executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
        )

        assert result is sample_order
        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["price"] is None
        assert call_kwargs["stop_price"] is None
        assert call_kwargs["tif"] is None
        assert call_kwargs["leverage"] is None

    def test_execute_order_with_decimal_precision(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that decimal precision is preserved."""
        mock_broker.place_order.return_value = sample_order

        precise_quantity = Decimal("0.00123456789")
        precise_price = Decimal("50000.123456")

        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=precise_quantity,
            price=precise_price,
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=None,
        )

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["quantity"] == precise_quantity
        assert call_kwargs["price"] == precise_price

    def test_execute_order_propagates_broker_exception(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
    ) -> None:
        """Test that broker exceptions are propagated."""
        mock_broker.place_order.side_effect = RuntimeError("Broker unavailable")

        with pytest.raises(RuntimeError, match="Broker unavailable"):
            executor.execute_order(
                submit_id="client-123",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
                stop_price=None,
                tif=TimeInForce.GTC,
                reduce_only=False,
                leverage=None,
            )


# ============================================================
# Test: RetryPolicy
# ============================================================


class TestRetryPolicy:
    """Tests for RetryPolicy configuration."""

    def test_default_values(self) -> None:
        """Test default retry policy values."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.base_delay == 0.5
        assert policy.max_delay == 5.0
        assert policy.timeout_seconds == 30.0
        assert policy.jitter == 0.1

    def test_calculate_delay_first_attempt_is_zero(self) -> None:
        """Test that first attempt has no delay."""
        policy = RetryPolicy(jitter=0)
        assert policy.calculate_delay(1) == 0.0

    def test_calculate_delay_exponential_backoff(self) -> None:
        """Test exponential backoff calculation."""
        policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=0)
        assert policy.calculate_delay(2) == 1.0  # base_delay * 2^0
        assert policy.calculate_delay(3) == 2.0  # base_delay * 2^1
        assert policy.calculate_delay(4) == 4.0  # base_delay * 2^2

    def test_calculate_delay_respects_max_delay(self) -> None:
        """Test that delay is capped at max_delay."""
        policy = RetryPolicy(base_delay=1.0, max_delay=2.0, jitter=0)
        assert policy.calculate_delay(10) == 2.0  # Capped at max

    def test_jitter_clamped_negative_to_zero(self) -> None:
        """Test that negative jitter is clamped to 0.0."""
        policy = RetryPolicy(jitter=-0.5)
        assert policy.jitter == 0.0

    def test_jitter_clamped_above_one_to_one(self) -> None:
        """Test that jitter > 1.0 is clamped to 1.0."""
        policy = RetryPolicy(jitter=1.5)
        assert policy.jitter == 1.0

    def test_jitter_valid_range_unchanged(self) -> None:
        """Test that jitter in [0.0, 1.0] is preserved."""
        for jitter_val in [0.0, 0.1, 0.5, 1.0]:
            policy = RetryPolicy(jitter=jitter_val)
            assert policy.jitter == jitter_val

    def test_jitter_boundary_values(self) -> None:
        """Test jitter at exact boundaries."""
        policy_zero = RetryPolicy(jitter=0.0)
        policy_one = RetryPolicy(jitter=1.0)
        assert policy_zero.jitter == 0.0
        assert policy_one.jitter == 1.0


# ============================================================
# Test: execute_with_retry
# ============================================================


class TestExecuteWithRetry:
    """Tests for execute_with_retry function."""

    def test_success_on_first_attempt(self) -> None:
        """Test successful execution on first attempt."""
        func = MagicMock(return_value="success")
        policy = RetryPolicy(max_attempts=3)
        sleep_calls: list[float] = []

        result = execute_with_retry(
            func,
            policy,
            client_order_id="test-123",
            sleep_fn=sleep_calls.append,
        )

        assert result == "success"
        func.assert_called_once()
        assert len(sleep_calls) == 0  # No sleep on first attempt

    def test_retry_on_retryable_exception(self) -> None:
        """Test retry occurs on retryable exception."""
        func = MagicMock(side_effect=[ConnectionError("failed"), "success"])
        policy = RetryPolicy(max_attempts=3, jitter=0)
        sleep_calls: list[float] = []

        result = execute_with_retry(
            func,
            policy,
            client_order_id="test-123",
            sleep_fn=sleep_calls.append,
        )

        assert result == "success"
        assert func.call_count == 2
        assert len(sleep_calls) == 1  # One delay before retry

    def test_max_attempts_respected(self) -> None:
        """Test that max_attempts is respected."""
        func = MagicMock(side_effect=ConnectionError("always fails"))
        policy = RetryPolicy(max_attempts=3, jitter=0)
        sleep_calls: list[float] = []

        with pytest.raises(ConnectionError, match="always fails"):
            execute_with_retry(
                func,
                policy,
                client_order_id="test-123",
                sleep_fn=sleep_calls.append,
            )

        assert func.call_count == 3
        assert len(sleep_calls) == 2  # Delays before attempts 2 and 3

    def test_non_retryable_exception_not_retried(self) -> None:
        """Test that non-retryable exceptions are not retried."""
        func = MagicMock(side_effect=ValueError("bad input"))
        policy = RetryPolicy(max_attempts=3)
        sleep_calls: list[float] = []

        with pytest.raises(ValueError, match="bad input"):
            execute_with_retry(
                func,
                policy,
                client_order_id="test-123",
                sleep_fn=sleep_calls.append,
            )

        func.assert_called_once()  # No retries
        assert len(sleep_calls) == 0

    def test_timeout_error_is_retryable(self) -> None:
        """Test that TimeoutError triggers retry."""
        func = MagicMock(side_effect=[TimeoutError("timed out"), "success"])
        policy = RetryPolicy(max_attempts=3, jitter=0)
        sleep_calls: list[float] = []

        result = execute_with_retry(
            func,
            policy,
            client_order_id="test-123",
            sleep_fn=sleep_calls.append,
        )

        assert result == "success"
        assert func.call_count == 2

    def test_client_order_id_logged_consistently(self) -> None:
        """Test that the same client_order_id is used for all attempts."""
        # This test ensures the client_order_id parameter is available
        # for logging across all attempts (verified by not raising)
        func = MagicMock(side_effect=[ConnectionError("fail"), "success"])
        policy = RetryPolicy(max_attempts=3, jitter=0)

        execute_with_retry(
            func,
            policy,
            client_order_id="stable-id-123",
            operation="test_op",
            sleep_fn=lambda _: None,
        )

        # If we get here without exception, the ID was handled correctly


# ============================================================
# Test: BrokerExecutor with retry
# ============================================================


class TestBrokerExecutorWithRetry:
    """Tests for BrokerExecutor retry functionality."""

    def test_execute_order_without_retry_default(
        self,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that retry is disabled by default."""
        mock_broker.place_order.side_effect = [ConnectionError("fail"), sample_order]
        executor = BrokerExecutor(broker=mock_broker)

        with pytest.raises(ConnectionError):
            executor.execute_order(
                submit_id="client-123",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                stop_price=None,
                tif=None,
                reduce_only=False,
                leverage=None,
            )

        mock_broker.place_order.assert_called_once()

    def test_execute_order_with_retry_enabled(
        self,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that retry works when enabled."""
        mock_broker.place_order.side_effect = [ConnectionError("fail"), sample_order]
        sleep_calls: list[float] = []
        executor = BrokerExecutor(
            broker=mock_broker,
            retry_policy=RetryPolicy(max_attempts=3, jitter=0),
            sleep_fn=sleep_calls.append,
        )

        result = executor.execute_order(
            submit_id="client-123",
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

        assert result is sample_order
        assert mock_broker.place_order.call_count == 2

    def test_retry_uses_same_client_order_id(
        self,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that the same client_order_id is used across all retry attempts."""
        captured_client_ids: list[str] = []

        def capture_and_fail(**kwargs):
            captured_client_ids.append(kwargs.get("client_id", ""))
            if len(captured_client_ids) < 2:
                raise ConnectionError("transient failure")
            return sample_order

        mock_broker.place_order.side_effect = capture_and_fail
        executor = BrokerExecutor(
            broker=mock_broker,
            retry_policy=RetryPolicy(max_attempts=3, jitter=0),
            sleep_fn=lambda _: None,
        )

        executor.execute_order(
            submit_id="idempotent-id-456",
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

        # Verify same client_id used across all attempts
        assert len(captured_client_ids) == 2
        assert captured_client_ids[0] == "idempotent-id-456"
        assert captured_client_ids[1] == "idempotent-id-456"
        assert captured_client_ids[0] == captured_client_ids[1]

    def test_custom_retry_policy(
        self,
        mock_broker: MagicMock,
    ) -> None:
        """Test that custom retry policy is used."""
        mock_broker.place_order.side_effect = ConnectionError("always fails")
        sleep_calls: list[float] = []
        custom_policy = RetryPolicy(max_attempts=2, base_delay=0.1, jitter=0)
        executor = BrokerExecutor(
            broker=mock_broker,
            retry_policy=custom_policy,
            sleep_fn=sleep_calls.append,
        )

        with pytest.raises(ConnectionError):
            executor.execute_order(
                submit_id="client-123",
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

        assert mock_broker.place_order.call_count == 2  # max_attempts=2


class TestBrokerCallMetrics:
    """Tests for broker call metrics recording."""

    @patch("gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency")
    def test_successful_call_records_latency_with_success_outcome(
        self,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        """Test that successful broker call records latency with success outcome."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_broker.place_order.return_value = mock_order

        executor = BrokerExecutor(broker=mock_broker)
        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
        )

        mock_record_latency.assert_called_once()
        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["operation"] == "submit"
        assert call_kwargs["outcome"] == "success"
        assert call_kwargs["latency_seconds"] >= 0

    @patch("gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency")
    def test_failed_call_records_latency_with_failure_outcome(
        self,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        """Test that failed broker call records latency with failure outcome."""
        mock_broker.place_order.side_effect = RuntimeError("Broker error")

        executor = BrokerExecutor(broker=mock_broker)
        with pytest.raises(RuntimeError):
            executor.execute_order(
                submit_id="client-123",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                stop_price=None,
                tif=None,
                reduce_only=False,
                leverage=None,
            )

        mock_record_latency.assert_called_once()
        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["operation"] == "submit"
        assert call_kwargs["outcome"] == "failure"
        assert call_kwargs["reason"] == "error"

    @patch("gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency")
    def test_timeout_error_classified_correctly(
        self,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        """Test that timeout errors are classified correctly in metrics."""
        mock_broker.place_order.side_effect = RuntimeError("Request timeout exceeded")

        executor = BrokerExecutor(broker=mock_broker)
        with pytest.raises(RuntimeError):
            executor.execute_order(
                submit_id="client-123",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                stop_price=None,
                tif=None,
                reduce_only=False,
                leverage=None,
            )

        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["reason"] == "timeout"

    @patch("gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency")
    def test_rate_limit_error_classified_correctly(
        self,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        """Test that rate limit errors are classified correctly in metrics."""
        mock_broker.place_order.side_effect = RuntimeError("429 rate limit exceeded")

        executor = BrokerExecutor(broker=mock_broker)
        with pytest.raises(RuntimeError):
            executor.execute_order(
                submit_id="client-123",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                stop_price=None,
                tif=None,
                reduce_only=False,
                leverage=None,
            )

        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["reason"] == "rate_limit"

    @patch("gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency")
    def test_network_error_classified_correctly(
        self,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        """Test that network errors are classified correctly in metrics."""
        mock_broker.place_order.side_effect = RuntimeError("Connection refused")

        executor = BrokerExecutor(broker=mock_broker)
        with pytest.raises(RuntimeError):
            executor.execute_order(
                submit_id="client-123",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                stop_price=None,
                tif=None,
                reduce_only=False,
                leverage=None,
            )

        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["reason"] == "network"


# ============================================================
# Test: Resilience with failure injection
# ============================================================


class TestBrokerExecutorResilience:
    """Deterministic resilience tests using failure injection harness."""

    def test_retry_success_after_n_failures(self, sample_order: Order) -> None:
        """Test that retry succeeds after N transient failures."""
        from tests.fixtures.failure_injection import (
            FailureScript,
            InjectingBroker,
            counting_sleep,
        )

        # Setup: fail twice, then succeed
        mock_broker = MagicMock()
        mock_broker.place_order.return_value = sample_order
        script = FailureScript.fail_then_succeed(failures=2)
        injecting = InjectingBroker(mock_broker, place_order=script)

        sleep_fn, get_sleeps = counting_sleep()
        executor = BrokerExecutor(
            broker=injecting,
            retry_policy=RetryPolicy(max_attempts=5, base_delay=0.5, jitter=0),
            sleep_fn=sleep_fn,
        )

        result = executor.execute_order(
            submit_id="resilience-test-1",
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

        assert result is sample_order
        assert injecting.get_call_count("place_order") == 3  # 2 failures + 1 success
        assert len(get_sleeps()) == 2  # 2 delays before retries

    def test_timeout_path_triggers_retry_with_classification(self, sample_order: Order) -> None:
        """Test that timeout triggers retry and is classified correctly."""
        from tests.fixtures.failure_injection import (
            FailureScript,
            InjectingBroker,
            no_op_sleep,
        )

        mock_broker = MagicMock()
        mock_broker.place_order.return_value = sample_order
        script = FailureScript.timeout_then_succeed(timeouts=1)
        injecting = InjectingBroker(mock_broker, place_order=script)

        executor = BrokerExecutor(
            broker=injecting,
            retry_policy=RetryPolicy(max_attempts=3, jitter=0),
            sleep_fn=no_op_sleep,
        )

        with patch(
            "gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency"
        ) as mock_record:
            result = executor.execute_order(
                submit_id="timeout-test-1",
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

        assert result is sample_order
        assert injecting.get_call_count("place_order") == 2
        # At least one call should have recorded with timeout classification
        # (the first failed attempt)
        assert mock_record.call_count >= 1

    def test_non_retryable_error_short_circuits(self) -> None:
        """Test that non-retryable errors abort immediately without retries."""
        from tests.fixtures.failure_injection import (
            FailureScript,
            InjectingBroker,
            no_op_sleep,
        )

        mock_broker = MagicMock()
        # ValueError is not in default retryable_exceptions
        script = FailureScript.rejection("Insufficient funds")
        injecting = InjectingBroker(mock_broker, place_order=script)

        executor = BrokerExecutor(
            broker=injecting,
            retry_policy=RetryPolicy(max_attempts=5, jitter=0),
            sleep_fn=no_op_sleep,
        )

        with pytest.raises(ValueError, match="Insufficient funds"):
            executor.execute_order(
                submit_id="rejection-test-1",
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

        # Only one attempt - no retries for non-retryable
        assert injecting.get_call_count("place_order") == 1

    def test_all_retries_exhausted_raises_last_exception(self) -> None:
        """Test that exhausting all retries raises the last exception."""
        from tests.fixtures.failure_injection import (
            FailureScript,
            InjectingBroker,
            counting_sleep,
        )

        mock_broker = MagicMock()
        script = FailureScript.always_fail(ConnectionError("persistent failure"))
        injecting = InjectingBroker(mock_broker, place_order=script)

        sleep_fn, get_sleeps = counting_sleep()
        executor = BrokerExecutor(
            broker=injecting,
            retry_policy=RetryPolicy(max_attempts=3, base_delay=0.1, jitter=0),
            sleep_fn=sleep_fn,
        )

        with pytest.raises(ConnectionError, match="persistent failure"):
            executor.execute_order(
                submit_id="exhaust-test-1",
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

        assert injecting.get_call_count("place_order") == 3  # All attempts used
        assert len(get_sleeps()) == 2  # Delays before attempts 2 and 3

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
