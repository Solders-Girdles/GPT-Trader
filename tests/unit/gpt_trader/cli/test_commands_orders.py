from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass

import pytest

import gpt_trader.cli.commands.orders as orders_cmd


def _make_args(**overrides):
    defaults = dict(
        profile="dev",
        orders_command="preview",
        symbol="BTC-PERP",
        side="buy",
        type="limit",
        quantity="0.5",
        price="42000",
        stop="41000",
        tif="IOC",
        client_id="client-1",
        leverage=3,
        reduce_only=True,
        order_id="abc",
        preview_id="def",
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_build_order_payload_includes_optional_fields():
    args = _make_args()
    payload = orders_cmd._build_order_payload(args)

    assert payload["symbol"] == "BTC-PERP"
    assert payload["side"].name == "BUY"
    assert payload["order_type"].name == "LIMIT"
    assert str(payload["quantity"]) == "0.5"
    assert payload["tif"].name == "IOC"
    assert str(payload["price"]) == "42000"
    assert str(payload["stop_price"]) == "41000"
    assert payload["reduce_only"] is True
    assert payload["leverage"] == 3
    assert payload["client_id"] == "client-1"


def test_build_order_payload_omits_optional_fields_when_missing():
    args = _make_args(
        type="market",
        price=None,
        stop=None,
        tif=None,
    )
    payload = orders_cmd._build_order_payload(args)

    assert "price" not in payload
    assert "stop_price" not in payload
    assert payload["tif"].name == "GTC"


def test_handle_preview_prints_json(monkeypatch, capsys):
    captured: dict[str, object] = {}

    def fake_build_config(args, *, skip):
        captured["skip"] = set(skip)
        return "config"

    class StubBroker:
        def preview_order(self, **payload):
            captured["payload"] = payload
            return {"preview": True}

        def edit_order_preview(self, order_id, **payload):
            raise AssertionError("Not expected in preview test")

        def edit_order(self, order_id, preview_id, **payload):
            raise AssertionError("Not expected in preview test")

    class StubBot:
        def __init__(self):
            self.broker = StubBroker()

        async def shutdown(self):
            captured["shutdown"] = True

    monkeypatch.setattr(orders_cmd.services, "build_config_from_args", fake_build_config)
    monkeypatch.setattr(orders_cmd.services, "instantiate_bot", lambda config: StubBot())

    exit_code = orders_cmd._handle_preview(_make_args())

    assert exit_code == 0
    assert captured["payload"]["symbol"] == "BTC-PERP"
    assert "orders_command" in captured["skip"]
    assert captured["shutdown"] is True
    out = capsys.readouterr().out
    assert json.loads(out)["preview"] is True


def test_handle_preview_errors_without_preview_support(monkeypatch):
    class StubBot:
        broker = object()

        async def shutdown(self):
            StubBot.shutdown_called = True

    StubBot.shutdown_called = False

    monkeypatch.setattr(orders_cmd.services, "build_config_from_args", lambda *_, **__: "config")
    monkeypatch.setattr(orders_cmd.services, "instantiate_bot", lambda config: StubBot())

    with pytest.raises(RuntimeError):
        orders_cmd._handle_preview(_make_args())

    assert StubBot.shutdown_called is False


def test_handle_edit_preview_errors_without_support(monkeypatch):
    class StubBot:
        broker = object()

        async def shutdown(self):
            StubBot.shutdown_called = True

    StubBot.shutdown_called = False

    monkeypatch.setattr(orders_cmd.services, "build_config_from_args", lambda *_, **__: "config")
    monkeypatch.setattr(orders_cmd.services, "instantiate_bot", lambda config: StubBot())

    with pytest.raises(RuntimeError):
        orders_cmd._handle_edit_preview(_make_args())

    assert StubBot.shutdown_called is False


def test_handle_edit_preview_invokes_broker(monkeypatch, capsys):
    class StubBroker:
        def preview_order(self, **kwargs):
            raise AssertionError("Not expected in edit preview test")

        def edit_order_preview(self, order_id, **payload):
            StubBroker.order_id = order_id
            StubBroker.payload = payload
            return {"edit": True}

        def edit_order(self, order_id, preview_id, **payload):
            raise AssertionError("Not expected in edit preview test")

    class StubBot:
        def __init__(self):
            self.broker = StubBroker()

        async def shutdown(self):
            StubBot.shutdown_called = True

    StubBot.shutdown_called = False

    monkeypatch.setattr(orders_cmd.services, "build_config_from_args", lambda *_, **__: "config")
    monkeypatch.setattr(orders_cmd.services, "instantiate_bot", lambda config: StubBot())

    exit_code = orders_cmd._handle_edit_preview(_make_args())

    assert exit_code == 0
    assert StubBot.shutdown_called is True
    assert StubBroker.order_id == "abc"
    assert StubBroker.payload["symbol"] == "BTC-PERP"
    out = capsys.readouterr().out
    assert json.loads(out)["edit"] is True


def test_handle_apply_edit_serializes_dataclass(monkeypatch, capsys):
    @dataclass
    class OrderResult:
        order_id: str
        status: str

    class StubBroker:
        def preview_order(self, **kwargs):
            raise AssertionError("Not expected in apply edit test")

        def edit_order_preview(self, order_id, **payload):
            raise AssertionError("Not expected in apply edit test")

        def edit_order(self, order_id, preview_id):
            return OrderResult(order_id=order_id, status=f"preview:{preview_id}")

    class StubBot:
        def __init__(self):
            self.broker = StubBroker()

        async def shutdown(self):
            StubBot.shutdown_called = True

    StubBot.shutdown_called = False

    monkeypatch.setattr(orders_cmd.services, "build_config_from_args", lambda *_, **__: "config")
    monkeypatch.setattr(orders_cmd.services, "instantiate_bot", lambda config: StubBot())

    exit_code = orders_cmd._handle_apply_edit(_make_args())

    assert exit_code == 0
    assert StubBot.shutdown_called is True
    out_data = json.loads(capsys.readouterr().out)
    assert out_data == {"order_id": "abc", "status": "preview:def"}


def test_handle_apply_edit_errors_without_support(monkeypatch):
    class StubBot:
        broker = object()

        async def shutdown(self):
            StubBot.shutdown_called = True

    StubBot.shutdown_called = False

    monkeypatch.setattr(orders_cmd.services, "build_config_from_args", lambda *_, **__: "config")
    monkeypatch.setattr(orders_cmd.services, "instantiate_bot", lambda config: StubBot())

    with pytest.raises(RuntimeError):
        orders_cmd._handle_apply_edit(_make_args())

    assert StubBot.shutdown_called is False


def test_handle_apply_edit_propagates_broker_errors(monkeypatch):
    class StubBroker:
        def preview_order(self, **kwargs):
            raise AssertionError("Not expected in failure test")

        def edit_order_preview(self, order_id, **payload):
            raise AssertionError("Not expected in failure test")

        def edit_order(self, order_id, preview_id):
            raise ValueError("broker failure")

    class StubBot:
        def __init__(self):
            self.broker = StubBroker()
            self.shutdown_called = False

        async def shutdown(self):
            self.shutdown_called = True

    bot = StubBot()

    monkeypatch.setattr(orders_cmd.services, "build_config_from_args", lambda *_, **__: "config")
    monkeypatch.setattr(orders_cmd.services, "instantiate_bot", lambda config: bot)

    with pytest.raises(ValueError):
        orders_cmd._handle_apply_edit(_make_args())

    assert bot.shutdown_called is True
