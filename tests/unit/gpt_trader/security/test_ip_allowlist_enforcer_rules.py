"""Tests for IPAllowlistEnforcer rule management."""

import pytest

from gpt_trader.security.ip_allowlist_enforcer import IPAllowlistEnforcer


@pytest.fixture
def enforcer(monkeypatch):
    """IP Allowlist Enforcer instance."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setenv("IP_ALLOWLIST_COINBASE_INTX", "192.168.1.1,10.0.0.0/24")
    return IPAllowlistEnforcer(enable_enforcement=True)


def test_load_rules_from_environment(enforcer):
    """Test loading rules from environment variables."""
    rules = enforcer.list_rules()
    assert len(rules) > 0

    coinbase_rule = enforcer.get_rule("coinbase_intx")
    assert coinbase_rule is not None
    assert "192.168.1.1" in coinbase_rule.allowed_ips
    assert "10.0.0.0/24" in coinbase_rule.allowed_ips


def test_add_rule(enforcer):
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


def test_add_rule_invalid_ip(enforcer):
    """Test adding rule with invalid IP."""
    success = enforcer.add_rule(
        "test_service",
        ["192.168.1.1", "invalid-ip"],
    )

    assert not success


def test_enable_disable_rule(enforcer):
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


def test_remove_rule(enforcer):
    """Test removing rule."""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    success = enforcer.remove_rule("test_service")
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule is None


def test_list_rules(enforcer):
    """Test listing all rules."""
    enforcer.add_rule("service1", ["192.168.1.1"])
    enforcer.add_rule("service2", ["10.0.0.0/24"])

    rules = enforcer.list_rules()
    assert len(rules) >= 2

    service_names = {r.service_name for r in rules}
    assert "service1" in service_names
    assert "service2" in service_names


def test_update_existing_rule_with_description(enforcer):
    """Test updating existing rule updates description."""
    enforcer.add_rule("test_service", ["192.168.1.1"], description="Original")

    success = enforcer.add_rule("test_service", ["192.168.1.2"], description="Updated description")
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule.description == "Updated description"
    assert "192.168.1.2" in rule.allowed_ips
    assert "192.168.1.1" not in rule.allowed_ips


def test_update_existing_rule_without_description(enforcer):
    """Test updating existing rule preserves description if not provided."""
    enforcer.add_rule("test_service", ["192.168.1.1"], description="Original")

    success = enforcer.add_rule("test_service", ["192.168.1.2"])
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule.description == "Original"


def test_enable_rule_nonexistent(enforcer):
    """Test enabling nonexistent rule returns False."""
    success = enforcer.enable_rule("nonexistent_service")
    assert not success


def test_disable_rule_nonexistent(enforcer):
    """Test disabling nonexistent rule returns False."""
    success = enforcer.disable_rule("nonexistent_service")
    assert not success


def test_remove_rule_nonexistent(enforcer):
    """Test removing nonexistent rule returns False."""
    success = enforcer.remove_rule("nonexistent_service")
    assert not success


def test_environment_empty_allowlist(monkeypatch):
    """Test that empty IP list from environment is skipped."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setenv("IP_ALLOWLIST_EMPTY_SERVICE", "")  # Empty value
    monkeypatch.setenv("IP_ALLOWLIST_VALID_SERVICE", "192.168.1.1")
    enforcer = IPAllowlistEnforcer(enable_enforcement=True)

    assert enforcer.get_rule("empty_service") is None
    assert enforcer.get_rule("valid_service") is not None
