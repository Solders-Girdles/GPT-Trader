"""Factory helpers for building order policy matrices."""

from __future__ import annotations

from decimal import Decimal

from .matrix import OrderPolicyMatrix


def create_standard_policy_matrix(environment: str = "sandbox") -> OrderPolicyMatrix:
    matrix = OrderPolicyMatrix(environment=environment)

    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
    increments = {
        "BTC-USD": {"size": Decimal("0.001"), "price": Decimal("1")},
        "ETH-USD": {"size": Decimal("0.01"), "price": Decimal("0.1")},
        "SOL-USD": {"size": Decimal("0.1"), "price": Decimal("0.001")},
        "XRP-USD": {"size": Decimal("1"), "price": Decimal("0.0001")},
    }

    for symbol in symbols:
        increment = increments.get(symbol, {"size": Decimal("0.001"), "price": Decimal("0.01")})
        matrix.add_symbol(
            symbol=symbol,
            min_order_size=increment["size"],
            size_increment=increment["size"],
            price_increment=increment["price"],
            max_order_size=Decimal("1000"),
            trading_enabled=True,
            reduce_only_mode=False,
            spread_threshold_bps=Decimal("20"),
        )

    return matrix


async def create_order_policy_matrix(environment: str = "sandbox") -> OrderPolicyMatrix:
    return create_standard_policy_matrix(environment=environment)


__all__ = ["create_standard_policy_matrix", "create_order_policy_matrix"]
