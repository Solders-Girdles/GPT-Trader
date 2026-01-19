"""Tests for IPAllowlistEnforcer internals and convenience APIs."""

import pytest

import gpt_trader.security.ip_allowlist_enforcer as ip_allowlist_mod
from gpt_trader.security.ip_allowlist_enforcer import IPAllowlistEnforcer


@pytest.fixture
def enforcer(monkeypatch):
    """IP Allowlist Enforcer instance."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setenv("IP_ALLOWLIST_COINBASE_INTX", "192.168.1.1,10.0.0.0/24")
    return IPAllowlistEnforcer(enable_enforcement=True)


def test_check_ip_cidr_edge_cases(enforcer):
    """Test CIDR edge cases."""
    assert enforcer._check_ip_in_allowlist("192.168.1.1", ["192.168.1.1/32"]) == "192.168.1.1/32"
    assert enforcer._check_ip_in_allowlist("192.168.1.1", ["0.0.0.0/0"]) == "0.0.0.0/0"
    assert enforcer._check_ip_in_allowlist("invalid-ip", ["192.168.1.0/24"]) is None


def test_check_ip_invalid_cidr_fallback(enforcer):
    """Test IP check with invalid CIDR that falls back to IP comparison."""
    result = enforcer._check_ip_in_allowlist("192.168.1.1", ["not-a-cidr", "192.168.1.1"])
    assert result == "192.168.1.1"


def test_check_ip_completely_invalid_entry(enforcer):
    """Test IP check skips completely invalid entries."""
    result = enforcer._check_ip_in_allowlist("192.168.1.1", ["completely-invalid", "also-invalid"])
    assert result is None


def test_global_singleton_and_convenience_functions(monkeypatch):
    """Test global singleton and convenience functions."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setattr(ip_allowlist_mod, "_ip_allowlist_enforcer", None)

    enforcer = ip_allowlist_mod.get_ip_allowlist_enforcer()
    assert enforcer is not None

    success = ip_allowlist_mod.add_ip_allowlist_rule("convenience_test", ["192.168.1.1"])
    assert success

    result = ip_allowlist_mod.validate_ip("192.168.1.1", "convenience_test")
    assert result is not None
