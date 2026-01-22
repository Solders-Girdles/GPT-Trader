from __future__ import annotations

import pytest

from gpt_trader.security.ip_allowlist_enforcer import IPAllowlistEnforcer


@pytest.fixture
def enforcer(monkeypatch: pytest.MonkeyPatch) -> IPAllowlistEnforcer:
    """IP Allowlist Enforcer instance."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setenv("IP_ALLOWLIST_COINBASE_INTX", "192.168.1.1,10.0.0.0/24")
    return IPAllowlistEnforcer(enable_enforcement=True)
