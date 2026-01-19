from __future__ import annotations

from argparse import Namespace


def make_args(**overrides):
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
