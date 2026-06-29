"""Integration tests exercising OrderEventRecorder against the real monitoring logger.

The recorder records order events through ``get_monitoring_logger()`` (a
``StructuredLogger``). These tests deliberately do **not** mock that logger so
the real ``log_event`` / ``log_order_submission`` / ``log_order_status_change``
methods are exercised. They guard against the regression where those methods did
not exist on ``StructuredLogger``, raising ``AttributeError`` at runtime (which
the best-effort wrappers swallowed into noisy exception tracebacks).
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder
from tests.unit.gpt_trader.utilities.logging_patterns_test_helpers import captured_logger


@pytest.fixture
def recorder_logger_spy(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Spy on the recorder's own module logger.

    On a monitoring-logger failure the recorder swallows the exception and
    records it via ``logger.exception("Failed to log ...")``. Asserting that was
    never called proves the real monitoring path ran without ``AttributeError``.
    """
    spy = MagicMock()
    monkeypatch.setattr(recorder_module, "logger", spy)
    return spy


def _assert_no_logging_failures(spy: MagicMock) -> None:
    assert spy.exception.call_count == 0, (
        "Recorder fell back to logger.exception, indicating the monitoring "
        "logger call failed (e.g. AttributeError)"
    )


def test_record_submission_attempt_uses_real_logger(
    order_event_recorder: OrderEventRecorder,
    recorder_logger_spy: MagicMock,
) -> None:
    with captured_logger("system") as (_, handler):
        order_event_recorder.record_submission_attempt(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
        )

    _assert_no_logging_failures(recorder_logger_spy)
    record = next(
        r for r in handler.records if getattr(r, "event_type", None) == "order_submission"
    )
    assert record.client_order_id == "client-123"  # type: ignore[attr-defined]
    assert record.symbol == "BTC-USD"  # type: ignore[attr-defined]
    assert record.side == "BUY"  # type: ignore[attr-defined]
    assert record.order_type == "LIMIT"  # type: ignore[attr-defined]


def test_record_rejection_uses_real_logger(
    order_event_recorder: OrderEventRecorder,
    recorder_logger_spy: MagicMock,
) -> None:
    with captured_logger("system") as (_, handler):
        order_event_recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
            client_order_id="client-123",
        )

    _assert_no_logging_failures(recorder_logger_spy)
    record = next(
        r for r in handler.records if getattr(r, "event_type", None) == "order_status_change"
    )
    assert record.to_status == "REJECTED"  # type: ignore[attr-defined]
    assert record.client_order_id == "client-123"  # type: ignore[attr-defined]


def test_record_trade_event_uses_real_logger(
    order_event_recorder: OrderEventRecorder,
    order_event_mock_order: MagicMock,
    recorder_logger_spy: MagicMock,
) -> None:
    with captured_logger("system") as (_, handler):
        order_event_recorder.record_trade_event(
            order=order_event_mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            submit_id="client-123",
        )

    _assert_no_logging_failures(recorder_logger_spy)
    record = next(
        r for r in handler.records if getattr(r, "event_type", None) == "order_status_change"
    )
    assert record.order_id == "order-123"  # type: ignore[attr-defined]


def test_record_preview_uses_real_logger(
    order_event_recorder: OrderEventRecorder,
    recorder_logger_spy: MagicMock,
) -> None:
    with captured_logger("system") as (_, handler):
        order_event_recorder.record_preview(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
            price=None,
            preview={"estimated_fee": "0.1"},
        )

    _assert_no_logging_failures(recorder_logger_spy)
    record = next(r for r in handler.records if getattr(r, "event_type", None) == "order_preview")
    assert record.component == "TradingEngine"  # type: ignore[attr-defined]
    assert record.symbol == "ETH-USD"  # type: ignore[attr-defined]
    assert record.side == "SELL"  # type: ignore[attr-defined]
