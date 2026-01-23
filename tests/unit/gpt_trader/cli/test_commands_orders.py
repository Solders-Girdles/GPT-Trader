from __future__ import annotations

import json

import pytest

import gpt_trader.cli.commands.orders as orders_cmd
from tests.unit.gpt_trader.cli.orders_command_test_helpers import make_args


def test_build_order_payload_includes_optional_fields():
    args = make_args()
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
    args = make_args(
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

    exit_code = orders_cmd._handle_preview(make_args())

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
        orders_cmd._handle_preview(make_args())

    assert StubBot.shutdown_called is False
