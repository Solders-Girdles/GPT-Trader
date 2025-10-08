from __future__ import annotations

from typing import Any

from bot_v2.monitoring.system import set_correlation_id
from src.bot_v2.features.brokerages.coinbase.client.base import CoinbaseClientBase


class DummyAuth:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def sign(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, str]:
        self.calls.append((method, path, body or {}))
        return {"CB-ACCESS-SIGN": "signature"}


def test_correlation_header_does_not_affect_signature() -> None:
    auth = DummyAuth()
    client = CoinbaseClientBase(
        base_url="https://api.exchange.test",
        auth=auth,
        enable_keep_alive=False,
    )

    captured_headers: dict[str, str] = {}

    def fake_transport(
        method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: int
    ) -> tuple[int, dict[str, str], str]:
        captured_headers.update(headers)
        return 200, {}, "{}"

    client._transport = fake_transport  # type: ignore[assignment]

    set_correlation_id("test-correlation-id")
    client._request("GET", "/api/v3/brokerage/accounts")

    assert auth.calls == [("GET", "/api/v3/brokerage/accounts", {})]
    assert captured_headers["X-Correlation-Id"] == "test-correlation-id"
