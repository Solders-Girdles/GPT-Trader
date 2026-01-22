"""Tests for IPAllowlistEnforcer rules and internals."""

from __future__ import annotations

import pytest

import gpt_trader.security.ip_allowlist_enforcer as ip_allowlist_mod
from gpt_trader.security.ip_allowlist_enforcer import IPAllowlistEnforcer


def test_check_ip_cidr_edge_cases(enforcer: IPAllowlistEnforcer) -> None:
    """Test CIDR edge cases."""
    assert enforcer._check_ip_in_allowlist("192.168.1.1", ["192.168.1.1/32"]) == "192.168.1.1/32"
    assert enforcer._check_ip_in_allowlist("192.168.1.1", ["0.0.0.0/0"]) == "0.0.0.0/0"
    assert enforcer._check_ip_in_allowlist("invalid-ip", ["192.168.1.0/24"]) is None


def test_check_ip_invalid_cidr_fallback(enforcer: IPAllowlistEnforcer) -> None:
    """Test IP check with invalid CIDR that falls back to IP comparison."""
    result = enforcer._check_ip_in_allowlist("192.168.1.1", ["not-a-cidr", "192.168.1.1"])
    assert result == "192.168.1.1"


def test_check_ip_completely_invalid_entry(enforcer: IPAllowlistEnforcer) -> None:
    """Test IP check skips completely invalid entries."""
    result = enforcer._check_ip_in_allowlist("192.168.1.1", ["completely-invalid", "also-invalid"])
    assert result is None


def test_global_singleton_and_convenience_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test global singleton and convenience functions."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setattr(ip_allowlist_mod, "_ip_allowlist_enforcer", None)

    enforcer = ip_allowlist_mod.get_ip_allowlist_enforcer()
    assert enforcer is not None

    success = ip_allowlist_mod.add_ip_allowlist_rule("convenience_test", ["192.168.1.1"])
    assert success

    result = ip_allowlist_mod.validate_ip("192.168.1.1", "convenience_test")
    assert result is not None


def test_load_rules_from_environment(enforcer: IPAllowlistEnforcer) -> None:
    """Test loading rules from environment variables."""
    rules = enforcer.list_rules()
    assert len(rules) > 0

    coinbase_rule = enforcer.get_rule("coinbase_intx")
    assert coinbase_rule is not None
    assert "192.168.1.1" in coinbase_rule.allowed_ips
    assert "10.0.0.0/24" in coinbase_rule.allowed_ips


def test_add_rule(enforcer: IPAllowlistEnforcer) -> None:
    """Test adding IP allowlist rule."""
    success = enforcer.add_rule(
        "coinbase_production",
        ["192.168.1.100", "10.0.0.0/8"],
        description="Production API server IPs",
    )

    assert success

    rule = enforcer.get_rule("coinbase_production")
    assert rule is not None
    assert rule.service_name == "coinbase_production"
    assert "192.168.1.100" in rule.allowed_ips
    assert "10.0.0.0/8" in rule.allowed_ips
    assert rule.description == "Production API server IPs"


def test_add_rule_invalid_ip(enforcer: IPAllowlistEnforcer) -> None:
    """Test adding rule with invalid IP."""
    success = enforcer.add_rule(
        "test_service",
        ["192.168.1.1", "invalid-ip"],
    )

    assert not success


def test_enable_disable_rule(enforcer: IPAllowlistEnforcer) -> None:
    """Test enabling and disabling rules."""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    # Disable
    success = enforcer.disable_rule("test_service")
    assert success

    rule = enforcer.get_rule("test_service")
    assert not rule.enabled

    # Enable
    success = enforcer.enable_rule("test_service")
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule.enabled


def test_remove_rule(enforcer: IPAllowlistEnforcer) -> None:
    """Test removing rule."""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    success = enforcer.remove_rule("test_service")
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule is None


def test_list_rules(enforcer: IPAllowlistEnforcer) -> None:
    """Test listing all rules."""
    enforcer.add_rule("service1", ["192.168.1.1"])
    enforcer.add_rule("service2", ["10.0.0.0/24"])

    rules = enforcer.list_rules()
    assert len(rules) >= 2

    service_names = {r.service_name for r in rules}
    assert "service1" in service_names
    assert "service2" in service_names


def test_update_existing_rule_with_description(enforcer: IPAllowlistEnforcer) -> None:
    """Test updating existing rule updates description."""
    enforcer.add_rule("test_service", ["192.168.1.1"], description="Original")

    success = enforcer.add_rule("test_service", ["192.168.1.2"], description="Updated description")
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule.description == "Updated description"
    assert "192.168.1.2" in rule.allowed_ips
    assert "192.168.1.1" not in rule.allowed_ips


def test_update_existing_rule_without_description(enforcer: IPAllowlistEnforcer) -> None:
    """Test updating existing rule preserves description if not provided."""
    enforcer.add_rule("test_service", ["192.168.1.1"], description="Original")

    success = enforcer.add_rule("test_service", ["192.168.1.2"])
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule.description == "Original"


def test_enable_rule_nonexistent(enforcer: IPAllowlistEnforcer) -> None:
    """Test enabling nonexistent rule returns False."""
    success = enforcer.enable_rule("nonexistent_service")
    assert not success


def test_disable_rule_nonexistent(enforcer: IPAllowlistEnforcer) -> None:
    """Test disabling nonexistent rule returns False."""
    success = enforcer.disable_rule("nonexistent_service")
    assert not success


def test_remove_rule_nonexistent(enforcer: IPAllowlistEnforcer) -> None:
    """Test removing nonexistent rule returns False."""
    success = enforcer.remove_rule("nonexistent_service")
    assert not success


def test_environment_empty_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that empty IP list from environment is skipped."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setenv("IP_ALLOWLIST_EMPTY_SERVICE", "")  # Empty value
    monkeypatch.setenv("IP_ALLOWLIST_VALID_SERVICE", "192.168.1.1")
    enforcer = IPAllowlistEnforcer(enable_enforcement=True)

    assert enforcer.get_rule("empty_service") is None
    assert enforcer.get_rule("valid_service") is not None
