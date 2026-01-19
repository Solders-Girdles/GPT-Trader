"""Coinbase account client endpoint tests."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from tests.unit.gpt_trader.features.brokerages.coinbase.helpers import (
    ACCOUNT_ENDPOINT_CASES,
    _decode_body,
    make_client,
)

pytestmark = pytest.mark.endpoints


class TestCoinbaseAccountEndpoints:
    @pytest.mark.parametrize("case", ACCOUNT_ENDPOINT_CASES, ids=lambda c: c["id"])
    def test_client_account_endpoints(self, case: dict[str, Any]) -> None:
        client = make_client()
        recorded: dict[str, Any] = {}

        def transport(method, url, headers, body, timeout):
            recorded["method"] = method
            recorded["url"] = url
            recorded["body"] = body
            return 200, {}, json.dumps(case.get("response", {}))

        client.set_transport_for_testing(transport)

        result = getattr(client, case["method"])(*case.get("args", ()), **case.get("kwargs", {}))

        assert recorded["method"] == case["expected_method"]
        parsed = urlparse(recorded["url"])
        assert parsed.path.endswith(case["expected_path"])

        expected_query = case.get("expected_query")
        if expected_query is not None:
            assert parse_qs(parsed.query) == expected_query
        else:
            assert parsed.query in ("", None)

        expected_payload = case.get("expected_payload")
        if expected_payload is not None:
            assert _decode_body(recorded.get("body")) == expected_payload
        else:
            assert not recorded.get("body")

        expected_result = case.get("expected_result")
        if expected_result is not None:
            assert result == expected_result
