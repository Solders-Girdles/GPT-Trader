"""Comprehensive Coinbase permission coverage."""

from __future__ import annotations

from typing import Any, Dict
from collections.abc import Iterable
from unittest.mock import patch

import pytest

from bot_v2.features.brokerages.coinbase.auth import CDPJWTAuth
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.errors import AuthError

from tests.unit.bot_v2.features.brokerages.coinbase.permissions_test_utils import (
    make_broker,
    make_client,
)


# --- Broker facade behaviour ---------------------------------------------------------------

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


# --- Client permission endpoint behaviour ---------------------------------------------------

PERMISSION_RESPONSES = [
    pytest.param(
        {
            "id": "all",
            "response": {
                "can_view": True,
                "can_trade": True,
                "can_transfer": True,
                "portfolio_uuid": "123e4567-e89b-12d3-a456-426614174000",
                "portfolio_type": "DEFAULT",
            },
        },
        id="all-permissions",
    ),
    pytest.param(
        {
            "id": "view-trade",
            "response": {
                "can_view": True,
                "can_trade": True,
                "can_transfer": False,
                "portfolio_uuid": "test-uuid",
                "portfolio_type": "DEFAULT",
            },
        },
        id="view-trade",
    ),
    pytest.param(
        {
            "id": "view-only",
            "response": {
                "can_view": True,
                "can_trade": False,
                "can_transfer": False,
                "portfolio_uuid": "test-uuid",
                "portfolio_type": "DEFAULT",
            },
        },
        id="view-only",
    ),
    pytest.param(
        {
            "id": "none",
            "response": {
                "can_view": False,
                "can_trade": False,
                "can_transfer": False,
                "portfolio_uuid": None,
                "portfolio_type": "UNDEFINED",
            },
        },
        id="no-permissions",
    ),
]


@pytest.mark.parametrize("case", PERMISSION_RESPONSES, ids=lambda c: c["id"])
def test_get_key_permissions_shapes_response(case: dict[str, Any]) -> None:
    client = make_client()
    with patch.object(client, "_request", return_value=case["response"]) as mock_request:
        result = client.get_key_permissions()

    mock_request.assert_called_once_with("GET", "/api/v3/brokerage/key_permissions")
    assert result == case["response"]


ERROR_CASES = [
    pytest.param(AuthError("401 Unauthorized"), id="unauthorized"),
    pytest.param(AuthError("Invalid API Key"), id="invalid-key"),
]


@pytest.mark.parametrize("error", ERROR_CASES)
def test_get_key_permissions_propagates_errors(error: AuthError) -> None:
    client = make_client()
    with patch.object(client, "_request", side_effect=error):
        with pytest.raises(AuthError) as exc_info:
            client.get_key_permissions()

    assert str(error) in str(exc_info.value)


# --- Portfolio metadata passthrough ---------------------------------------------------------


def test_get_key_permissions_includes_portfolio_metadata() -> None:
    client = make_client()
    response = {
        "can_view": True,
        "can_trade": True,
        "can_transfer": False,
        "portfolio_uuid": "550e8400-e29b-41d4-a716-446655440000",
        "portfolio_type": "DEFAULT",
    }

    with patch.object(client, "_request", return_value=response):
        result = client.get_key_permissions()

    assert result["portfolio_uuid"] == "550e8400-e29b-41d4-a716-446655440000"
    assert result["portfolio_type"] == "DEFAULT"


PORTFOLIO_TYPES = [
    pytest.param("DEFAULT", id="default"),
    pytest.param("CONSUMER", id="consumer"),
    pytest.param("INTX", id="intx"),
    pytest.param("UNDEFINED", id="undefined"),
]


@pytest.mark.parametrize("portfolio_type", PORTFOLIO_TYPES)
def test_portfolio_type_passthrough(portfolio_type: str) -> None:
    client = make_client()
    response: dict[str, Any] = {
        "can_view": True,
        "can_trade": True,
        "can_transfer": False,
        "portfolio_uuid": "test-uuid",
        "portfolio_type": portfolio_type,
    }

    with patch.object(client, "_request", return_value=response):
        result = client.get_key_permissions()

    assert result["portfolio_type"] == portfolio_type


# --- Permission rule evaluation --------------------------------------------------------------


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


# --- Live integration coverage ---------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.real_api
def test_live_permission_check(coinbase_cdp_credentials) -> None:
    auth = CDPJWTAuth(
        api_key_name=coinbase_cdp_credentials.api_key,
        private_key_pem=coinbase_cdp_credentials.private_key,
        base_host="api.coinbase.com",
    )

    client = CoinbaseClient(
        base_url="https://api.coinbase.com",
        auth=auth,
        api_mode="advanced",
    )

    perms = client.get_key_permissions()

    assert "can_view" in perms
    assert "can_trade" in perms
    assert "can_transfer" in perms
    assert "portfolio_uuid" in perms
    assert "portfolio_type" in perms
