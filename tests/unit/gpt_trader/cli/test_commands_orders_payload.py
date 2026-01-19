from __future__ import annotations

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
