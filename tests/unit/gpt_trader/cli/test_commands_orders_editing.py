from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

import gpt_trader.cli.commands.orders as orders_cmd
from tests.unit.gpt_trader.cli.orders_command_test_helpers import make_args


def test_handle_edit_preview_errors_without_support(monkeypatch):
    class StubBot:
        broker = object()

        async def shutdown(self):
            StubBot.shutdown_called = True

    StubBot.shutdown_called = False

    monkeypatch.setattr(orders_cmd.services, "build_config_from_args", lambda *_, **__: "config")
    monkeypatch.setattr(orders_cmd.services, "instantiate_bot", lambda config: StubBot())

    with pytest.raises(RuntimeError):
        orders_cmd._handle_edit_preview(make_args())

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

    exit_code = orders_cmd._handle_edit_preview(make_args())

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

    exit_code = orders_cmd._handle_apply_edit(make_args())

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
        orders_cmd._handle_apply_edit(make_args())

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
        orders_cmd._handle_apply_edit(make_args())

    assert bot.shutdown_called is True
