import json
from types import SimpleNamespace

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient, CoinbaseAuth


def test_auth_inherits_client_api_mode_and_json_body(monkeypatch):
    # Auth without explicit api_mode, but client is advanced
    auth = CoinbaseAuth(api_key="k", api_secret="s", passphrase="p")
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")

    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(SimpleNamespace(method=method, url=url, headers=headers, body=body, timeout=timeout))
        return 200, {"content-type": "application/json"}, json.dumps({"ok": True})

    client.set_transport_for_testing(fake_transport)

    # Force a POST with no body to verify we send {} and sign consistently
    client._request("POST", "/api/v3/brokerage/orders", body=None)  # type: ignore[arg-type]

    # Validate one call captured
    assert len(calls) == 1
    c = calls[0]

    # Since client.api_mode is advanced, auth should not include passphrase and should use integer timestamp
    h = c.headers
    assert "CB-ACCESS-KEY" in h
    assert "CB-ACCESS-SIGN" in h
    assert "CB-ACCESS-TIMESTAMP" in h
    # No passphrase when using advanced mode (even if auth was constructed with one)
    assert "CB-ACCESS-PASSPHRASE" not in h

    # Body should be JSON '{}'
    assert isinstance(c.body, (bytes, bytearray))
    assert c.body.decode("utf-8") == "{}"

