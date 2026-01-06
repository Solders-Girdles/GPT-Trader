from __future__ import annotations

import pytest

from gpt_trader.core import AuthError, BrokerageError
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient


def _make_client() -> CoinbaseClient:
    # auth is optional for tests; disable keep-alive to avoid urllib opener setup.
    return CoinbaseClient(base_url="https://api.test", auth=None, enable_keep_alive=False)


def _patch_sleep(mocker) -> None:
    mocker.patch("gpt_trader.features.brokerages.coinbase.client.base.time.sleep", lambda _s: None)


def test_request_retries_on_429_and_5xx_then_succeeds(mocker) -> None:
    client = _make_client()
    _patch_sleep(mocker)
    responses = [
        (429, {"retry-after": "0.05"}, '{"message": "rate limit"}'),
        (502, {}, '{"message": "server error"}'),
        (200, {}, '{"ok": true}'),
    ]

    def fake_transport(
        method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: int
    ) -> tuple[int, dict[str, str], str]:
        status, hdrs, text = responses.pop(0)
        return status, hdrs, text

    client._transport = fake_transport  # type: ignore[assignment]

    result = client._request("GET", "/accounts")

    assert result == {"ok": True}
    assert responses == []


def test_request_fast_fails_on_auth_errors(mocker) -> None:
    client = _make_client()
    _patch_sleep(mocker)

    def transport(
        method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: int
    ) -> tuple[int, dict[str, str], str]:
        return 401, {}, '{"message": "invalid api key"}'

    client._transport = transport  # type: ignore[assignment]

    with pytest.raises(AuthError) as exc_info:
        client._request("GET", "/accounts")

    assert "invalid" in str(exc_info.value).lower()


def test_non_json_response_yields_brokerage_error(mocker) -> None:
    client = _make_client()
    _patch_sleep(mocker)

    def transport(
        method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: int
    ) -> tuple[int, dict[str, str], str]:
        return 500, {}, "<html>oops</html>"

    client._transport = transport  # type: ignore[assignment]

    with pytest.raises(BrokerageError) as exc_info:
        client._request("GET", "/boom")

    assert "unknown" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()
