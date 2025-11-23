"""Tests for IP Allowlist Enforcer"""

from unittest.mock import Mock

import pytest

from bot_v2.security.ip_allowlist_enforcer import (
    IPAllowlistEnforcer,
)


@pytest.fixture
def enforcer():
    """IP Allowlist Enforcer instance"""
    settings = Mock()
    settings.raw_env = {
        "IP_ALLOWLIST_ENABLED": "1",
        "IP_ALLOWLIST_COINBASE_INTX": "192.168.1.1,10.0.0.0/24",
    }
    return IPAllowlistEnforcer(settings=settings)


def test_load_rules_from_environment(enforcer):
    """Test loading rules from environment variables"""
    rules = enforcer.list_rules()
    assert len(rules) > 0

    coinbase_rule = enforcer.get_rule("coinbase_intx")
    assert coinbase_rule is not None
    assert "192.168.1.1" in coinbase_rule.allowed_ips
    assert "10.0.0.0/24" in coinbase_rule.allowed_ips


def test_add_rule(enforcer):
    """Test adding IP allowlist rule"""
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
    """Test adding rule with invalid IP"""
    success = enforcer.add_rule(
        "test_service",
        ["192.168.1.1", "invalid-ip"],
    )

    assert not success


def test_validate_ip_exact_match(enforcer):
    """Test IP validation with exact match"""
    enforcer.add_rule("test_service", ["192.168.1.1", "192.168.1.2"])

    result = enforcer.validate_ip("192.168.1.1", "test_service")

    assert result.is_allowed
    assert result.matched_rule == "192.168.1.1"


def test_validate_ip_cidr_match(enforcer):
    """Test IP validation with CIDR match"""
    enforcer.add_rule("test_service", ["10.0.0.0/24"])

    result = enforcer.validate_ip("10.0.0.50", "test_service")

    assert result.is_allowed
    assert result.matched_rule == "10.0.0.0/24"


def test_validate_ip_not_in_allowlist(enforcer):
    """Test IP validation rejection"""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    result = enforcer.validate_ip("10.0.0.1", "test_service")

    assert not result.is_allowed
    assert result.matched_rule is None
    assert "not in allowlist" in result.reason


def test_validate_ip_no_rule(enforcer):
    """Test IP validation with no rule configured"""
    result = enforcer.validate_ip("192.168.1.1", "nonexistent_service")

    assert not result.is_allowed
    assert "No IP allowlist rule" in result.reason


def test_validate_ip_disabled_rule(enforcer):
    """Test IP validation with disabled rule"""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.disable_rule("test_service")

    result = enforcer.validate_ip("192.168.1.1", "test_service")

    assert not result.is_allowed
    assert "disabled" in result.reason


def test_enable_disable_rule(enforcer):
    """Test enabling and disabling rules"""
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
    """Test removing rule"""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    success = enforcer.remove_rule("test_service")
    assert success

    rule = enforcer.get_rule("test_service")
    assert rule is None


def test_enforcement_disabled():
    """Test with enforcement disabled"""
    settings = Mock()
    settings.raw_env = {"IP_ALLOWLIST_ENABLED": "0"}

    enforcer = IPAllowlistEnforcer(settings=settings)

    # Should allow all IPs when enforcement is disabled
    result = enforcer.validate_ip("any-ip", "any-service")
    assert result.is_allowed
    assert "disabled" in result.reason


def test_validation_logging(enforcer):
    """Test validation logging"""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    # Generate some validations
    enforcer.validate_ip("192.168.1.1", "test_service")
    enforcer.validate_ip("10.0.0.1", "test_service")

    log = enforcer.get_validation_log(limit=10)
    assert len(log) == 2

    # Check log entries
    assert log[0]["client_ip"] == "192.168.1.1"
    assert log[0]["is_allowed"] is True

    assert log[1]["client_ip"] == "10.0.0.1"
    assert log[1]["is_allowed"] is False


def test_clear_validation_log(enforcer):
    """Test clearing validation log"""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.validate_ip("192.168.1.1", "test_service")

    enforcer.clear_validation_log()

    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_check_ip_cidr_edge_cases(enforcer):
    """Test CIDR edge cases"""
    # /32 CIDR (single IP)
    assert enforcer._check_ip_in_allowlist("192.168.1.1", ["192.168.1.1/32"]) == "192.168.1.1/32"

    # /0 CIDR (all IPs)
    assert enforcer._check_ip_in_allowlist("192.168.1.1", ["0.0.0.0/0"]) == "0.0.0.0/0"

    # Invalid IP
    assert enforcer._check_ip_in_allowlist("invalid-ip", ["192.168.1.0/24"]) is None


def test_list_rules(enforcer):
    """Test listing all rules"""
    enforcer.add_rule("service1", ["192.168.1.1"])
    enforcer.add_rule("service2", ["10.0.0.0/24"])

    rules = enforcer.list_rules()
    assert len(rules) >= 2

    service_names = {r.service_name for r in rules}
    assert "service1" in service_names
    assert "service2" in service_names
