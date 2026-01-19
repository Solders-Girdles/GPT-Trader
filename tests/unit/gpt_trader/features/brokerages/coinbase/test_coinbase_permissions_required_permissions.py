"""Tests for permission rule evaluation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import patch

import pytest

from tests.unit.gpt_trader.features.brokerages.coinbase.permissions_test_utils import (  # naming: allow
    make_client,
)


def _check_required_permissions(
    permissions: dict[str, bool], required: Iterable[str]
) -> tuple[bool, list[str]]:
    """Return whether all required permissions are present and list missing ones."""
    missing = [name for name in required if not permissions.get(name, False)]
    return not missing, missing


ALL_GRANTED_CASES = [
    pytest.param(
        {"can_view": True, "can_trade": True, "can_transfer": True}, ["can_view"], id="view"
    ),
    pytest.param(
        {"can_view": True, "can_trade": True, "can_transfer": True},
        ["can_view", "can_trade"],
        id="view-trade",
    ),
    pytest.param(
        {"can_view": True, "can_trade": True, "can_transfer": True},
        ["can_view", "can_trade", "can_transfer"],
        id="all",
    ),
]


@pytest.mark.parametrize("permissions, required", ALL_GRANTED_CASES)
def test_check_required_permissions_all_granted(
    permissions: dict[str, bool], required: Iterable[str]
) -> None:
    all_granted, missing = _check_required_permissions(permissions, required)
    assert all_granted is True
    assert missing == []


MISSING_CASES = [
    pytest.param(
        {"can_view": True, "can_trade": False, "can_transfer": False},
        ["can_view", "can_trade"],
        ["can_trade"],
        id="missing-trade",
    ),
    pytest.param(
        {"can_view": True, "can_trade": False, "can_transfer": False},
        ["can_transfer"],
        ["can_transfer"],
        id="missing-transfer",
    ),
    pytest.param(
        {"can_view": False, "can_trade": False, "can_transfer": False},
        ["can_view", "can_trade", "can_transfer"],
        ["can_view", "can_trade", "can_transfer"],
        id="missing-all",
    ),
]


@pytest.mark.parametrize("permissions, required, expected_missing", MISSING_CASES)
def test_check_required_permissions_missing(
    permissions: dict[str, bool], required: Iterable[str], expected_missing: list[str]
) -> None:
    all_granted, missing = _check_required_permissions(permissions, required)
    assert all_granted is False
    assert missing == expected_missing


def test_stubbed_permission_response_satisfies_required_keys() -> None:
    client = make_client()
    response = {
        "can_view": True,
        "can_trade": True,
        "can_transfer": False,
        "portfolio_uuid": "stub-uuid",
        "portfolio_type": "DEFAULT",
    }

    with patch.object(client, "_request", return_value=response):
        permissions = client.get_key_permissions()

    required_keys = ("can_view", "can_trade", "can_transfer", "portfolio_uuid", "portfolio_type")
    for key in required_keys:
        assert key in permissions

    granted, missing = _check_required_permissions(
        {key: permissions[key] for key in ("can_view", "can_trade", "can_transfer")},
        ("can_view", "can_trade"),
    )
    assert granted is True
    assert missing == []
    assert permissions["portfolio_uuid"] == "stub-uuid"
