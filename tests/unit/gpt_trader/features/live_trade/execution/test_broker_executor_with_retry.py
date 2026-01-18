"""Tests for BrokerExecutor retry functionality."""

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
