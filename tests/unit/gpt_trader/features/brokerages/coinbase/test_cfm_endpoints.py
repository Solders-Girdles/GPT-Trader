"""Unit tests for CoinbaseClient CFM (perpetuals) endpoints.

Covers cfm_balance_summary, cfm_positions, cfm_position, cfm_intraday_* methods.
"""

import json

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient

pytestmark = pytest.mark.endpoints


def make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def test_cfm_balance_summary_requires_advanced_mode():
    """Test cfm_balance_summary is only available in advanced mode."""
    client = make_client(api_mode="advanced")
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"balance": "10000", "margin": "5000"})

    client.set_transport_for_testing(fake_transport)
    out = client.cfm_balance_summary()

    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/cfm/balance_summary")
    assert "balance" in out
    assert "margin" in out


def test_cfm_positions_formats_path():
    """Test cfm_positions endpoint path."""
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"positions": [{"symbol": "BTC-PERP", "size": "1.5"}]})

    client.set_transport_for_testing(fake_transport)
    out = client.cfm_positions()

    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/cfm/positions")
    assert "positions" in out


def test_cfm_position_with_product_id():
    """Test cfm_position with specific product_id."""
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"position": {"symbol": "ETH-PERP", "size": "10"}})

    client.set_transport_for_testing(fake_transport)
    out = client.cfm_position("ETH-PERP")

    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/cfm/positions/ETH-PERP")
    assert "position" in out


def test_cfm_intraday_current_margin_window():
    """Test cfm_intraday_current_margin_window endpoint."""
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"margin_window": "INTRADAY_HIGH_MARGIN_1H"})

    client.set_transport_for_testing(fake_transport)
    out = client.cfm_intraday_current_margin_window()

    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/cfm/intraday/current_margin_window")
    assert out["margin_window"] == "INTRADAY_HIGH_MARGIN_1H"


def test_cfm_intraday_position_refresh_not_implemented():
    """Test cfm_intraday_position_refresh endpoint - currently not implemented."""
    # This endpoint is not yet implemented in CoinbaseClient
    # When implemented, it should:
    # - Send POST to /api/v3/brokerage/cfm/intraday/position_refresh
    # - Return refresh status
    assert True  # Placeholder until method is implemented


def test_cfm_sweep_not_implemented():
    """Test cfm_sweep endpoint - currently not implemented."""
    # This endpoint is not yet implemented in CoinbaseClient
    # When implemented, it should:
    # - Send POST to /api/v3/brokerage/cfm/sweeps
    # - Include from_account and to_account in payload
    # - Return sweep status with sweep_id
    assert True  # Placeholder until method is implemented


def test_cfm_methods_require_advanced_mode():
    """Test that CFM methods are not available in exchange mode."""
    from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError

    client_ex = make_client(api_mode="exchange")

    # CFM methods should raise InvalidRequestError in exchange mode
    try:
        client_ex.cfm_balance_summary()
        assert False, "Expected InvalidRequestError for CFM in exchange mode"
    except InvalidRequestError as e:
        assert "not available in exchange mode" in str(e)


def test_cfm_sweeps_and_schedule_paths():
    """Test cfm_sweeps and cfm_sweeps_schedule endpoints paths (GET)."""
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url))
        # Return minimal payloads
        if url.endswith("/api/v3/brokerage/cfm/sweeps"):
            return 200, {}, json.dumps({"sweeps": []})
        if url.endswith("/api/v3/brokerage/cfm/sweeps/schedule"):
            return 200, {}, json.dumps({"schedule": {}})
        return 200, {}, json.dumps({})

    client.set_transport_for_testing(transport)

    # Sweeps history
    out1 = client.cfm_sweeps()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/cfm/sweeps")
    assert "sweeps" in out1

    # Sweeps schedule
    out2 = client.cfm_sweeps_schedule()
    assert calls[1][0] == "GET"
    assert calls[1][1].endswith("/api/v3/brokerage/cfm/sweeps/schedule")
    assert "schedule" in out2


def test_cfm_intraday_margin_setting_posts_payload():
    client = make_client()
    captured = {}

    def transport(method, url, headers, body, timeout):
        captured["method"] = method
        captured["url"] = url
        captured["body"] = json.loads(body or b"{}")
        return 200, {}, json.dumps({"ok": True})

    client.set_transport_for_testing(transport)
    payload = {"margin_window": "8h"}
    out = client.cfm_intraday_margin_setting(payload)
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/api/v3/brokerage/cfm/intraday/margin_setting")
    assert captured["body"] == payload
    assert out["ok"] is True
