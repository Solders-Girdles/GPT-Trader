"""Tests for broker permission facade behavior."""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit.gpt_trader.features.brokerages.coinbase.permissions_test_utils import (  # naming: allow
    make_broker,
)

BROKER_PERMISSION_CASES = [
    pytest.param(
        {
            "id": "trading-allowed",
            "permissions": {
                "can_view": True,
                "can_trade": True,
                "can_transfer": False,
            },
            "expected": {
                "can_trade": True,
                "can_view": True,
            },
        },
        id="trading-allowed",
    ),
    pytest.param(
        {
            "id": "view-only",
            "permissions": {
                "can_view": True,
                "can_trade": False,
                "can_transfer": False,
            },
            "expected": {
                "can_trade": False,
                "can_view": True,
            },
        },
        id="view-only",
    ),
]


@pytest.mark.parametrize("case", BROKER_PERMISSION_CASES, ids=lambda c: c["id"])
def test_broker_permission_lookup(monkeypatch, case: dict[str, Any]) -> None:
    broker = make_broker()
    monkeypatch.setattr(broker.client, "get_key_permissions", lambda: case["permissions"])

    perms = broker.client.get_key_permissions()
    for key, expected in case["expected"].items():
        assert perms[key] is expected
