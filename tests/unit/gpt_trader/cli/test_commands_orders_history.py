from __future__ import annotations

from typing import Callable

from gpt_trader.cli.response import CliErrorCode, CliResponse
import gpt_trader.cli.commands.orders as orders_cmd
from gpt_trader.persistence.orders_store import OrderStatus
from tests.unit.gpt_trader.cli.orders_command_test_helpers import make_history_args
from tests.unit.gpt_trader.persistence.orders_store_test_helpers import create_test_order


class StubOrdersStore:
    def __init__(self, records: list, *, callback: Callable[..., None] | None = None) -> None:
        self.records = records
        self.callback = callback
        self.called_with: dict[str, object] | None = None

    def list_orders(
        self,
        *,
        limit: int,
        symbol: str | None,
        status: OrderStatus | None,
    ) -> list:
        self.called_with = {"limit": limit, "symbol": symbol, "status": status}
        if self.callback:
            self.callback(limit, symbol, status)
        return self.records


def _use_stub_store(monkeypatch, store: StubOrdersStore) -> None:
    monkeypatch.setattr(
        orders_cmd,
        "_with_orders_store",
        lambda args, callback: callback(store),
    )


def test_history_list_text_output(monkeypatch, capsys):
    record = create_test_order(order_id="text-order", status=OrderStatus.FILLED)
    store = StubOrdersStore([record])
    _use_stub_store(monkeypatch, store)

    args = make_history_args(limit=3, symbol="BTC-USD", status="filled", output_format="text")
    exit_code = orders_cmd._handle_history_list(args)

    assert exit_code == 0
    assert store.called_with == {
        "limit": 3,
        "symbol": "BTC-USD",
        "status": OrderStatus.FILLED,
    }

    output = capsys.readouterr().out
    assert "Order history" in output
    assert "text-order" in output
    assert "symbol=BTC-USD" in output
    assert "status=filled" in output


def test_history_list_json_response(monkeypatch):
    record = create_test_order(order_id="json-order", status=OrderStatus.OPEN)
    store = StubOrdersStore([record])
    _use_stub_store(monkeypatch, store)

    args = make_history_args(limit=2, output_format="json")
    response = orders_cmd._handle_history_list(args)

    assert isinstance(response, CliResponse)
    assert response.success
    assert response.data["count"] == 1
    assert response.data["orders"][0]["order_id"] == "json-order"
    assert response.data["filters"]["limit"] == 2
    assert store.called_with == {
        "limit": 2,
        "symbol": None,
        "status": None,
    }


def test_history_list_reports_empty(monkeypatch, capsys):
    store = StubOrdersStore([])
    _use_stub_store(monkeypatch, store)

    args = make_history_args()
    exit_code = orders_cmd._handle_history_list(args)

    assert exit_code == 0
    assert "No order history records found." in capsys.readouterr().out
    assert store.called_with == {
        "limit": orders_cmd._DEFAULT_HISTORY_LIMIT,
        "symbol": None,
        "status": None,
    }


def test_history_list_invalid_status_returns_error():
    args = make_history_args(status="unknown", output_format="json")
    response = orders_cmd._handle_history_list(args)

    assert isinstance(response, CliResponse)
    assert not response.success
    assert response.errors[0].code == CliErrorCode.INVALID_ARGUMENT.value
    assert response.errors[0].details == {"status": "unknown"}


def test_history_list_limit_validation():
    args = make_history_args(limit=0, output_format="json")
    response = orders_cmd._handle_history_list(args)

    assert isinstance(response, CliResponse)
    assert not response.success
    assert response.errors[0].code == CliErrorCode.INVALID_ARGUMENT.value
    assert response.errors[0].details == {"limit": 0}


def test_history_list_storage_error_returns_failure(monkeypatch):
    def raise_error(args, callback):
        raise RuntimeError("boom")

    monkeypatch.setattr(orders_cmd, "_with_orders_store", raise_error)
    response = orders_cmd._handle_history_list(make_history_args(output_format="json"))

    assert isinstance(response, CliResponse)
    assert not response.success
    assert response.errors[0].code == CliErrorCode.OPERATION_FAILED.value
    assert response.errors[0].details == {"error": "boom"}
