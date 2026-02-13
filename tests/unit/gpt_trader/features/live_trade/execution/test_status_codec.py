"""Status codec tests for execution-to-store translations."""

from __future__ import annotations

import pytest

from gpt_trader.features.live_trade.execution.status_codec import (
    ExecutionStatusCodecError,
    execution_status_for_event,
    execution_status_for_store,
)
from gpt_trader.persistence.orders_store import OrderStatus as StoreOrderStatus


@pytest.mark.parametrize(
    "input_status, expected",
    [
        ("partially_filled", StoreOrderStatus.PARTIALLY_FILLED),
        ("Cancelled", StoreOrderStatus.CANCELLED),
        ("REJECTED", StoreOrderStatus.REJECTED),
        ("retry", StoreOrderStatus.OPEN),
        ("retrying", StoreOrderStatus.OPEN),
    ],
)
def test_execution_status_for_store_maps_variants(
    input_status: str, expected: StoreOrderStatus
) -> None:
    assert execution_status_for_store(input_status) == expected


@pytest.mark.parametrize(
    "input_status, expected",
    [
        ("partial_fill", StoreOrderStatus.PARTIALLY_FILLED.value),
        ("cancelled", StoreOrderStatus.CANCELLED.value),
        ("rejected", StoreOrderStatus.REJECTED.value),
    ],
)
def test_execution_status_for_event_returns_store_value(input_status: str, expected: str) -> None:
    assert execution_status_for_event(input_status) == expected


def test_execution_status_for_store_raises_on_unknown() -> None:
    with pytest.raises(ExecutionStatusCodecError, match="Unsupported execution status"):
        execution_status_for_store("unmapped_status")
