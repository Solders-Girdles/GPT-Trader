"""Tests for Coinbase client key permission endpoint behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.brokerages.coinbase.errors import AuthError
from tests.unit.gpt_trader.features.brokerages.coinbase.permissions_test_utils import (  # naming: allow
    make_client,
)

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
def test_get_key_permissions_shapes_response(
    case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()
    mock_request = MagicMock(return_value=case["response"])
    monkeypatch.setattr(client, "_request", mock_request)

    result = client.get_key_permissions()

    mock_request.assert_called_once_with("GET", "/api/v3/brokerage/key_permissions")
    assert result == case["response"]


ERROR_CASES = [
    pytest.param(AuthError("401 Unauthorized"), id="unauthorized"),
    pytest.param(AuthError("Invalid API Key"), id="invalid-key"),
]


@pytest.mark.parametrize("error", ERROR_CASES)
def test_get_key_permissions_propagates_errors(
    error: AuthError,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()
    monkeypatch.setattr(client, "_request", MagicMock(side_effect=error))

    with pytest.raises(AuthError) as exc_info:
        client.get_key_permissions()

    assert str(error) in str(exc_info.value)


def test_get_key_permissions_includes_portfolio_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()
    response = {
        "can_view": True,
        "can_trade": True,
        "can_transfer": False,
        "portfolio_uuid": "550e8400-e29b-41d4-a716-446655440000",
        "portfolio_type": "DEFAULT",
    }
    monkeypatch.setattr(client, "_request", MagicMock(return_value=response))

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
def test_portfolio_type_passthrough(
    portfolio_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = make_client()
    response: dict[str, Any] = {
        "can_view": True,
        "can_trade": True,
        "can_transfer": False,
        "portfolio_uuid": "test-uuid",
        "portfolio_type": portfolio_type,
    }
    monkeypatch.setattr(client, "_request", MagicMock(return_value=response))

    result = client.get_key_permissions()

    assert result["portfolio_type"] == portfolio_type
