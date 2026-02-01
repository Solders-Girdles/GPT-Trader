"""Tests for order submission retry behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import gpt_trader.features.live_trade.execution.order_submission as order_submission_module
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


def test_submit_order_retries_transient_error_once(
    mock_broker: MagicMock,
    mock_event_store: MagicMock,
    open_orders: list[str],
    submit_order_with_result_call,
    mock_order,
) -> None:
    sleep_calls: list[float] = []

    submitter = OrderSubmitter(
        broker=mock_broker,
        event_store=mock_event_store,
        bot_id="test-bot",
        open_orders=open_orders,
        enable_retries=True,
        sleep_fn=sleep_calls.append,
    )
    mock_broker.place_order.side_effect = [ConnectionError("network down"), mock_order]

    outcome = submit_order_with_result_call(submitter, client_order_id="retry-123")

    assert outcome.success is True
    assert outcome.order_id == "order-123"
    assert mock_broker.place_order.call_count == 2
    assert len(sleep_calls) == 1
    delay = sleep_calls[0]
    assert delay >= order_submission_module.SUBMISSION_RETRY_POLICY.base_delay
    assert delay <= order_submission_module.SUBMISSION_RETRY_POLICY.max_delay


def test_submit_order_no_retry_on_non_transient_error(
    mock_broker: MagicMock,
    mock_event_store: MagicMock,
    open_orders: list[str],
    submit_order_with_result_call,
) -> None:
    sleep_calls: list[float] = []

    submitter = OrderSubmitter(
        broker=mock_broker,
        event_store=mock_event_store,
        bot_id="test-bot",
        open_orders=open_orders,
        enable_retries=True,
        sleep_fn=sleep_calls.append,
    )
    mock_broker.place_order.side_effect = ValueError("insufficient funds")

    outcome = submit_order_with_result_call(submitter, client_order_id="retry-456")

    assert outcome.failed is True
    assert mock_broker.place_order.call_count == 1
    assert sleep_calls == []
