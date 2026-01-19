"""Tests for IPAllowlistEnforcer IP validation behavior."""

import pytest

from gpt_trader.security.ip_allowlist_enforcer import IPAllowlistEnforcer


@pytest.fixture
def enforcer(monkeypatch):
    """IP Allowlist Enforcer instance."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "1")
    monkeypatch.setenv("IP_ALLOWLIST_COINBASE_INTX", "192.168.1.1,10.0.0.0/24")
    return IPAllowlistEnforcer(enable_enforcement=True)


def test_validate_ip_exact_match(enforcer):
    """Test IP validation with exact match."""
    enforcer.add_rule("test_service", ["192.168.1.1", "192.168.1.2"])

    result = enforcer.validate_ip("192.168.1.1", "test_service")

    assert result.is_allowed
    assert result.matched_rule == "192.168.1.1"


def test_validate_ip_cidr_match(enforcer):
    """Test IP validation with CIDR match."""
    enforcer.add_rule("test_service", ["10.0.0.0/24"])

    result = enforcer.validate_ip("10.0.0.50", "test_service")

    assert result.is_allowed
    assert result.matched_rule == "10.0.0.0/24"


def test_validate_ip_not_in_allowlist(enforcer):
    """Test IP validation rejection."""
    enforcer.add_rule("test_service", ["192.168.1.1"])

    result = enforcer.validate_ip("10.0.0.1", "test_service")

    assert not result.is_allowed
    assert result.matched_rule is None
    assert "not in allowlist" in result.reason


def test_validate_ip_no_rule(enforcer):
    """Test IP validation with no rule configured."""
    result = enforcer.validate_ip("192.168.1.1", "nonexistent_service")

    assert not result.is_allowed
    assert "No IP allowlist rule" in result.reason


def test_validate_ip_disabled_rule(enforcer):
    """Test IP validation with disabled rule."""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.disable_rule("test_service")

    result = enforcer.validate_ip("192.168.1.1", "test_service")

    assert not result.is_allowed
    assert "disabled" in result.reason


def test_enforcement_disabled(monkeypatch):
    """Test with enforcement disabled."""
    monkeypatch.setenv("IP_ALLOWLIST_ENABLED", "0")

    enforcer = IPAllowlistEnforcer(enable_enforcement=False)

    result = enforcer.validate_ip("any-ip", "any-service")
    assert result.is_allowed
    assert "disabled" in result.reason


def test_validate_ip_without_logging(enforcer):
    """Test validation with log_validation=False."""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.clear_validation_log()

    result = enforcer.validate_ip("192.168.1.1", "test_service", log_validation=False)

    assert result.is_allowed
    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_validate_ip_no_rule_without_logging(enforcer):
    """Test validation with no rule and log_validation=False."""
    enforcer.clear_validation_log()

    result = enforcer.validate_ip("192.168.1.1", "unknown_service", log_validation=False)

    assert not result.is_allowed
    log = enforcer.get_validation_log()
    assert len(log) == 0


def test_validate_ip_disabled_rule_without_logging(enforcer):
    """Test validation with disabled rule and log_validation=False."""
    enforcer.add_rule("test_service", ["192.168.1.1"])
    enforcer.disable_rule("test_service")
    enforcer.clear_validation_log()

    result = enforcer.validate_ip("192.168.1.1", "test_service", log_validation=False)

    assert not result.is_allowed
    log = enforcer.get_validation_log()
    assert len(log) == 0
